//! Extract function signatures and type definitions from source files
//! using tree-sitter parsers.
//!
//! Provides a structural extraction of function signatures, struct/class
//! definitions, trait definitions, and impl blocks from source code via
//! the tree-sitter family of grammar crates. This is used by the context
//! compression system (`tmg_core::context::ContextCompressor`) to replace
//! large `file_read` results with a structural summary instead of the
//! full file body.
//!
//! Supported languages:
//! - Rust (`.rs`)
//! - Python (`.py`)
//! - TypeScript (`.ts`, `.tsx`)
//! - JavaScript (`.js`, `.jsx`, `.mjs`, `.mts`, `.cjs`)
//!
//! # Determinism
//!
//! [`extract_signatures`] returns signatures sorted by `(line, kind)`,
//! so the output of two equal inputs is bit-identical regardless of
//! tree-sitter's internal pattern-execution order.

use std::ffi::OsStr;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::LazyLock;

use tree_sitter::{Language, Node, Parser, Query, QueryCursor, StreamingIterator as _};

/// A single extracted signature from a source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    /// The line number (1-based) where this signature starts.
    pub line: usize,
    /// The kind of definition (e.g., "fn", "struct", "class").
    pub kind: &'static str,
    /// The extracted signature text (header line, body stripped).
    pub text: String,
}

/// Detect language from a file extension and extract structural signatures.
///
/// Returns `None` if the file extension is not recognized as a supported
/// language. Returns `Some(vec![])` when the language *is* recognised but
/// no top-level definitions were found (e.g. empty source, parse error
/// suppressed by tree-sitter's error tolerance, or a runtime tree-sitter
/// ABI mismatch that produced an empty query). Callers that need to
/// distinguish "no symbols" from "internal extraction failure" should
/// use [`extract_signatures_detailed`].
#[must_use]
pub fn extract_signatures(filename: &str, source: &str) -> Option<Vec<Signature>> {
    let lang = Lang::from_filename(filename)?;
    Some(lang.extract(source).unwrap_or_default())
}

/// Errors returned by [`extract_signatures_detailed`] when a recognised
/// language fails to parse or when the per-language query was unable to
/// compile (almost always a tree-sitter grammar/ABI mismatch).
///
/// Returned to callers that want to log a warning instead of silently
/// falling back to tail-truncation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExtractError {
    /// The tree-sitter grammar's ABI is incompatible with the runtime
    /// crate version, so `Parser::set_language` rejected it.
    #[error("tree-sitter grammar ABI mismatch")]
    LanguageAbiMismatch,
    /// The compiled query for this language has zero patterns, which
    /// means the static query string failed to compile at startup.
    /// This indicates a bug in the query body or an ABI mismatch.
    #[error("tree-sitter query failed to compile")]
    QueryCompileFailed,
    /// `Parser::parse` returned `None` (tree-sitter typically only does
    /// this on cancellation / timeout / allocation failure).
    #[error("tree-sitter parser returned no tree")]
    ParseFailed,
}

/// Like [`extract_signatures`] but surfaces extraction failures so
/// callers can log a warning when a recognised language fails to parse.
///
/// Returns:
/// - `Some(Ok(vec))` on a successful extraction (possibly empty when
///   the source has no top-level definitions),
/// - `Some(Err(e))` when the language was recognised but extraction
///   failed (ABI mismatch, query compile failure, or a parse error),
/// - `None` when the file extension is not recognised.
#[must_use]
pub fn extract_signatures_detailed(
    filename: &str,
    source: &str,
) -> Option<Result<Vec<Signature>, ExtractError>> {
    let lang = Lang::from_filename(filename)?;
    Some(lang.extract(source))
}

/// Format extracted signatures into a condensed string summary.
///
/// The output shows line numbers and definition types, suitable for
/// inclusion in a compressed-context message.
#[must_use]
pub fn format_signatures(signatures: &[Signature]) -> String {
    let mut buf = String::new();
    for sig in signatures {
        let _ = writeln!(buf, "L{}: [{}] {}", sig.line, sig.kind, sig.text);
    }
    buf
}

// ---------------------------------------------------------------------------
// Language dispatch
// ---------------------------------------------------------------------------

/// One of the supported source languages.
#[derive(Debug, Clone, Copy)]
enum Lang {
    Rust,
    Python,
    TypeScript,
    Tsx,
    JavaScript,
}

impl Lang {
    fn from_filename(filename: &str) -> Option<Self> {
        // Use `Path::extension()` to handle dotfiles (e.g. `.bashrc`),
        // paths with dots in directory components (e.g.
        // `foo.bar/baz`), and extensionless files correctly.
        let ext = Path::new(filename).extension().and_then(OsStr::to_str)?;
        match ext {
            "rs" => Some(Self::Rust),
            "py" | "pyi" => Some(Self::Python),
            "ts" | "mts" | "cts" => Some(Self::TypeScript),
            "tsx" => Some(Self::Tsx),
            "js" | "jsx" | "mjs" | "cjs" => Some(Self::JavaScript),
            _ => None,
        }
    }

    fn ts_language(self) -> Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Self::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        }
    }

    /// Map a capture name back to a stable `Signature::kind` label.
    fn kind_for_capture(name: &str) -> Option<&'static str> {
        match name {
            "fn" => Some("fn"),
            "struct" | "enum" | "trait" | "type_alias" | "union" | "ts_type" | "ts_enum"
            | "ts_module" => Some("type"),
            "impl" => Some("impl"),
            "macro" => Some("macro"),
            "py_fn" => Some("def"),
            "py_class" | "class" => Some("class"),
            "function" | "arrow_fn" => Some("function"),
            "interface" => Some("interface"),
            _ => None,
        }
    }

    fn extract(self, source: &str) -> Result<Vec<Signature>, ExtractError> {
        let mut parser = Parser::new();
        if parser.set_language(&self.ts_language()).is_err() {
            return Err(ExtractError::LanguageAbiMismatch);
        }
        let Some(tree) = parser.parse(source, None) else {
            return Err(ExtractError::ParseFailed);
        };
        let query = match self {
            Self::Rust => &*RUST_QUERY_COMPILED,
            Self::Python => &*PYTHON_QUERY_COMPILED,
            Self::TypeScript => &*TS_QUERY_COMPILED,
            Self::Tsx => &*TSX_QUERY_COMPILED,
            Self::JavaScript => &*JS_QUERY_COMPILED,
        };

        // A zero-pattern query indicates the static query body failed to
        // compile (almost always an ABI / grammar mismatch). Surface that
        // instead of silently returning an empty Vec so callers can log
        // and fall back to tail-truncation with a clear root cause.
        if query.pattern_count() == 0 {
            return Err(ExtractError::QueryCompileFailed);
        }

        Ok(collect_from_query(
            query,
            &tree.root_node(),
            source.as_bytes(),
        ))
    }
}

/// Walk the captures produced by `query` over `root` and emit one
/// [`Signature`] per matched top-level definition.
///
/// We iterate `matches` (rather than `captures`) so each match yields
/// a single capture for the definition node we care about; capture
/// names tell us which `kind` to report.
fn collect_from_query(query: &Query, root: &Node<'_>, source: &[u8]) -> Vec<Signature> {
    let mut cursor = QueryCursor::new();
    let capture_names = query.capture_names();
    let mut sigs: Vec<Signature> = Vec::new();

    let mut matches = cursor.matches(query, *root, source);
    while let Some(m) = matches.next() {
        for cap in m.captures {
            let cap_name = capture_names
                .get(cap.index as usize)
                .copied()
                .unwrap_or_default();
            let Some(kind) = Lang::kind_for_capture(cap_name) else {
                continue;
            };
            let node = cap.node;
            let line = node.start_position().row.saturating_add(1);
            let text = signature_header(&node, source);
            // Skip empty headers (defensive — should not happen for valid
            // captures).
            if text.is_empty() {
                continue;
            }
            sigs.push(Signature { line, kind, text });
        }
    }

    // Tree-sitter does not guarantee order across multiple patterns;
    // sort by (line, kind) for a deterministic result.
    sigs.sort_by(|a, b| a.line.cmp(&b.line).then_with(|| a.kind.cmp(b.kind)));
    sigs.dedup_by(|a, b| a.line == b.line && a.kind == b.kind && a.text == b.text);
    sigs
}

/// Extract the "header" portion of a definition node — everything up to
/// the body block — as a single trimmed line.
///
/// For function-like nodes this is the signature line up to (but not
/// including) `{` / `:` / `=>`. For struct/enum/trait nodes it stops at
/// the body delimiter as well. Newlines inside the header (e.g. wrapped
/// argument lists) are collapsed to single spaces so the result is one
/// readable line per signature.
fn signature_header(node: &Node<'_>, source: &[u8]) -> String {
    let body_start = locate_body_start(node, source);
    let start = node.start_byte();
    let end = body_start.unwrap_or_else(|| node.end_byte());
    let end = end.clamp(start, source.len());
    let raw = std::str::from_utf8(&source[start..end]).unwrap_or("");
    collapse_whitespace(raw)
}

/// Find the byte offset where this definition's body starts so we can
/// strip it. We look for the first child node named `body`, `block`, or
/// (for impl/struct/trait) a node whose kind ends with `_block` /
/// `field_declaration_list` / `enum_variant_list`.
fn locate_body_start(node: &Node<'_>, source: &[u8]) -> Option<usize> {
    // Common field name across grammars.
    if let Some(body) = node.child_by_field_name("body") {
        return Some(body.start_byte());
    }
    // Walk children for known body node kinds.
    let mut walker = node.walk();
    for child in node.children(&mut walker) {
        let kind = child.kind();
        if matches!(
            kind,
            "block"
                | "declaration_list"
                | "field_declaration_list"
                | "enum_variant_list"
                | "class_body"
                | "interface_body"
                | "object_type"
                | "statement_block"
        ) {
            return Some(child.start_byte());
        }
    }
    // Fallback: if the source contains an opening brace, cut there.
    let start = node.start_byte();
    let end = node.end_byte();
    let slice = source.get(start..end)?;
    let rel_brace = slice.iter().position(|b| *b == b'{')?;
    Some(start + rel_brace)
}

fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_space && !out.is_empty() {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    // Trim trailing whitespace and any trailing punctuation that comes
    // right before a body delimiter we already cut.
    while matches!(out.chars().last(), Some(c) if c.is_whitespace() || c == '{' || c == '=') {
        out.pop();
    }
    out.trim().to_owned()
}

// ---------------------------------------------------------------------------
// Tree-sitter queries
// ---------------------------------------------------------------------------

/// Rust top-level definitions we surface in the structural summary.
///
/// **Note on intentional omissions**: module (`mod` items), constants
/// (`const`), and statics (`static`) are *not* extracted. The summary
/// is meant to expose the call surface (functions, types, traits,
/// impls, macros) rather than every binding; callers that need a
/// constant's value still see it in the tail-truncated fallback.
const RUST_QUERY: &str = r"
(function_item) @fn
(struct_item) @struct
(enum_item) @enum
(trait_item) @trait
(impl_item) @impl
(type_item) @type_alias
(union_item) @union
(macro_definition) @macro
";

const PYTHON_QUERY: &str = r"
(function_definition) @py_fn
(class_definition) @py_class
";

/// TypeScript captures.
///
/// Extends the base set with:
/// - `arrow_fn`: `const foo = (...) => {...}` style declarations,
///   which dominate modern TS code and were covered by the pre-#49
///   regex extractor.
/// - `ts_enum`: `enum X {...}` declarations.
/// - `ts_module`: `namespace`/`module` declarations (treated as
///   type-level grouping for the structural summary).
const TS_QUERY: &str = r"
(function_declaration) @function
(class_declaration) @class
(abstract_class_declaration) @class
(interface_declaration) @interface
(type_alias_declaration) @ts_type
(enum_declaration) @ts_enum
(internal_module) @ts_module
(module) @ts_module
(lexical_declaration
    (variable_declarator
        value: (arrow_function))) @arrow_fn
";

/// JavaScript captures.
///
/// Adds `arrow_fn` for `const foo = () => {...}` so modern JS code
/// (which rarely uses `function` declarations at the top level) still
/// produces a signature summary.
const JS_QUERY: &str = r"
(function_declaration) @function
(class_declaration) @class
(lexical_declaration
    (variable_declarator
        value: (arrow_function))) @arrow_fn
";

// Compile each query lazily once so repeated extract calls don't pay
// the parser-construction cost. Compilation failures (tree-sitter
// ABI mismatches and the like) collapse to an empty-body query so the
// `LazyLock` itself never panics — `Lang::extract` then returns
// [`ExtractError::QueryCompileFailed`] and the caller can log a
// warning and fall back to tail truncation.
static RUST_QUERY_COMPILED: LazyLock<Query> = LazyLock::new(|| {
    Query::new(&Lang::Rust.ts_language(), RUST_QUERY)
        .unwrap_or_else(|_| empty_query(&Lang::Rust.ts_language()))
});
static PYTHON_QUERY_COMPILED: LazyLock<Query> = LazyLock::new(|| {
    Query::new(&Lang::Python.ts_language(), PYTHON_QUERY)
        .unwrap_or_else(|_| empty_query(&Lang::Python.ts_language()))
});
static TS_QUERY_COMPILED: LazyLock<Query> = LazyLock::new(|| {
    Query::new(&Lang::TypeScript.ts_language(), TS_QUERY)
        .unwrap_or_else(|_| empty_query(&Lang::TypeScript.ts_language()))
});
static TSX_QUERY_COMPILED: LazyLock<Query> = LazyLock::new(|| {
    Query::new(&Lang::Tsx.ts_language(), TS_QUERY)
        .unwrap_or_else(|_| empty_query(&Lang::Tsx.ts_language()))
});
static JS_QUERY_COMPILED: LazyLock<Query> = LazyLock::new(|| {
    Query::new(&Lang::JavaScript.ts_language(), JS_QUERY)
        .unwrap_or_else(|_| empty_query(&Lang::JavaScript.ts_language()))
});

/// Build an empty-body `Query` for `language` as a panic-free fallback
/// when the real query body fails to compile.
///
/// Empty bodies always compile (tree-sitter accepts them as valid),
/// but on the chance that even this fails we fall back to a Rust empty
/// query — and if *that* also fails we hand back a structurally invalid
/// `Query` only in spirit: callers gate on `pattern_count() == 0` so a
/// bogus query yields zero matches without any panic.
fn empty_query(language: &Language) -> Query {
    Query::new(language, "")
        .or_else(|_| Query::new(&Lang::Rust.ts_language(), ""))
        // The very last fallback should be unreachable in practice;
        // tree-sitter always accepts an empty body. We still avoid
        // `unwrap`/`expect`/`unreachable!` inside a `LazyLock`
        // initialiser by using the only constructor available — if
        // `Query::new` legitimately can't produce *anything* the host
        // tree-sitter is broken and the binary will fail elsewhere on
        // the next parse anyway.
        .unwrap_or_else(|_| {
            // SAFETY-BY-CONSTRUCTION: `tree-sitter` accepts every
            // empty source string we have ever observed; the chained
            // fallback above already covered the only known failure
            // mode. We retry once more on the original language so
            // the LazyLock body always returns *some* Query value.
            #[expect(
                clippy::expect_used,
                reason = "panic only fires if tree-sitter itself is unable to construct any empty query, which would already break parsing elsewhere"
            )]
            Query::new(language, "").expect("tree-sitter cannot construct any empty query")
        })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[expect(
    clippy::expect_used,
    reason = "tests use expect to fail loudly when a supported extension unexpectedly returns None"
)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_function_signatures() {
        let source = r#"
pub fn hello(name: &str) -> String {
    format!("Hello, {name}")
}

async fn fetch_data(url: &str) -> Result<Data, Error> {
    todo!()
}

pub(crate) unsafe fn raw_ptr() -> *const u8 {
    std::ptr::null()
}
"#;
        let sigs = extract_signatures("lib.rs", source).expect("rust ext supported");
        assert!(sigs.len() >= 3, "expected >= 3 sigs, got {}", sigs.len());
        assert!(
            sigs.iter()
                .any(|s| s.kind == "fn" && s.text.contains("pub fn hello"))
        );
        assert!(
            sigs.iter()
                .any(|s| s.kind == "fn" && s.text.contains("async fn fetch_data"))
        );
        assert!(
            sigs.iter()
                .any(|s| s.kind == "fn" && s.text.contains("unsafe fn raw_ptr"))
        );
    }

    #[test]
    fn rust_type_definitions() {
        let source = r"
pub struct MyStruct {
    field: String,
}

enum Color {
    Red,
    Green,
    Blue,
}

pub trait Drawable {
    fn draw(&self);
}

impl Drawable for MyStruct {
    fn draw(&self) {}
}
";
        let sigs = extract_signatures("lib.rs", source).expect("rust ext supported");
        assert!(
            sigs.iter()
                .any(|s| s.kind == "type" && s.text.contains("struct MyStruct"))
        );
        assert!(
            sigs.iter()
                .any(|s| s.kind == "type" && s.text.contains("enum Color"))
        );
        assert!(
            sigs.iter()
                .any(|s| s.kind == "type" && s.text.contains("trait Drawable"))
        );
        assert!(
            sigs.iter()
                .any(|s| s.kind == "impl" && s.text.contains("impl Drawable for MyStruct"))
        );
    }

    #[test]
    fn rust_macro_definition() {
        let source = r"
macro_rules! my_macro {
    () => {};
}
";
        let sigs = extract_signatures("lib.rs", source).expect("rust ext supported");
        assert!(
            sigs.iter()
                .any(|s| s.kind == "macro" && s.text.contains("my_macro")),
            "expected macro signature, got: {sigs:?}"
        );
    }

    #[test]
    fn python_signatures() {
        let source = r"
class MyClass(BaseClass):
    def __init__(self, name: str) -> None:
        self.name = name

    async def fetch(self, url: str) -> dict:
        pass

def standalone_func(x: int, y: int) -> int:
    return x + y
";
        let sigs = extract_signatures("app.py", source).expect("python ext supported");
        assert!(
            sigs.iter()
                .any(|s| s.kind == "class" && s.text.contains("class MyClass"))
        );
        assert!(
            sigs.iter()
                .any(|s| s.kind == "def" && s.text.contains("__init__"))
        );
        assert!(
            sigs.iter()
                .any(|s| s.kind == "def" && s.text.contains("async def fetch"))
        );
        assert!(
            sigs.iter()
                .any(|s| s.kind == "def" && s.text.contains("def standalone_func"))
        );
    }

    #[test]
    fn typescript_signatures() {
        let source = r"
export function greet(name: string): string {
    return `Hello, ${name}`;
}

export async function fetchData(url: string): Promise<Data> {
    return await fetch(url);
}

export class UserService extends BaseService {
    constructor(private db: Database) {}
}

interface Config {
    host: string;
    port: number;
}

export type Result<T> = { ok: true; value: T } | { ok: false; error: Error };
";
        let sigs = extract_signatures("index.ts", source).expect("ts ext supported");
        assert!(
            sigs.iter()
                .any(|s| s.kind == "function" && s.text.contains("function greet"))
        );
        assert!(
            sigs.iter()
                .any(|s| s.kind == "function" && s.text.contains("async function fetchData"))
        );
        assert!(
            sigs.iter()
                .any(|s| s.kind == "class" && s.text.contains("class UserService"))
        );
        assert!(
            sigs.iter()
                .any(|s| s.kind == "interface" && s.text.contains("interface Config"))
        );
        assert!(
            sigs.iter()
                .any(|s| s.kind == "type" && s.text.contains("type Result"))
        );
    }

    #[test]
    fn javascript_signatures() {
        let source = r"
export function greet(name) {
    return `Hello, ${name}`;
}

export class UserService extends BaseService {
    constructor(db) {}
}
";
        let sigs = extract_signatures("app.js", source).expect("js ext supported");
        assert!(
            sigs.iter()
                .any(|s| s.kind == "function" && s.text.contains("function greet"))
        );
        assert!(
            sigs.iter()
                .any(|s| s.kind == "class" && s.text.contains("class UserService"))
        );
    }

    #[test]
    fn javascript_arrow_function() {
        let source = r"
const greet = (name) => `Hello, ${name}`;
export const compute = (x, y) => x + y;
";
        let sigs = extract_signatures("app.js", source).expect("js ext supported");
        assert!(
            sigs.iter().any(|s| s.kind == "function"
                && (s.text.contains("greet") || s.text.contains("const greet"))),
            "expected arrow `greet` signature, got: {sigs:?}"
        );
        assert!(
            sigs.iter().any(|s| s.kind == "function"
                && (s.text.contains("compute") || s.text.contains("const compute"))),
            "expected arrow `compute` signature, got: {sigs:?}"
        );
    }

    #[test]
    fn typescript_arrow_function() {
        let source = r"
export const greet = (name: string): string => `Hello, ${name}`;
const compute = (x: number, y: number) => x + y;
";
        let sigs = extract_signatures("index.ts", source).expect("ts ext supported");
        assert!(
            sigs.iter().any(|s| s.kind == "function"
                && (s.text.contains("greet") || s.text.contains("const greet"))),
            "expected arrow `greet` signature, got: {sigs:?}"
        );
        assert!(
            sigs.iter().any(|s| s.kind == "function"
                && (s.text.contains("compute") || s.text.contains("const compute"))),
            "expected arrow `compute` signature, got: {sigs:?}"
        );
    }

    #[test]
    fn typescript_enum_and_namespace() {
        let source = r"
export enum Color {
    Red,
    Green,
    Blue,
}

namespace Geometry {
    export interface Point { x: number; y: number; }
}
";
        let sigs = extract_signatures("index.ts", source).expect("ts ext supported");
        assert!(
            sigs.iter()
                .any(|s| s.kind == "type" && s.text.contains("enum Color")),
            "expected ts enum signature, got: {sigs:?}"
        );
        assert!(
            sigs.iter()
                .any(|s| s.kind == "type" && s.text.contains("Geometry")),
            "expected namespace/module signature, got: {sigs:?}"
        );
    }

    #[test]
    fn typescript_abstract_class() {
        let source = r"
export abstract class Shape {
    abstract area(): number;
}
";
        let sigs = extract_signatures("index.ts", source).expect("ts ext supported");
        assert!(
            sigs.iter()
                .any(|s| s.kind == "class" && s.text.contains("class Shape")),
            "expected abstract class signature, got: {sigs:?}"
        );
    }

    #[test]
    fn tsx_signatures() {
        let source = r"
export function App(): JSX.Element {
    return <div>hi</div>;
}
";
        let sigs = extract_signatures("App.tsx", source).expect("tsx ext supported");
        assert!(
            sigs.iter()
                .any(|s| s.kind == "function" && s.text.contains("function App"))
        );
    }

    #[test]
    fn detect_language_from_extension() {
        assert!(extract_signatures("main.rs", "fn main() {}").is_some());
        assert!(extract_signatures("app.py", "def main(): pass").is_some());
        assert!(extract_signatures("index.ts", "function main() {}").is_some());
        assert!(extract_signatures("app.tsx", "function App() {}").is_some());
        assert!(extract_signatures("app.js", "function main() {}").is_some());
        assert!(extract_signatures("data.json", "{}").is_none());
        assert!(extract_signatures("README.md", "# hi").is_none());
    }

    /// `Path::extension()` correctly skips dotfiles, treats
    /// extensionless names as having no extension, and only consults
    /// the final path component (not directory names).
    #[test]
    fn detect_language_uses_path_extension() {
        // Dotfiles: ".bashrc" has no extension under POSIX semantics.
        assert!(extract_signatures(".bashrc", "echo hi").is_none());
        // Extensionless basename, even when surrounded by dotted dirs.
        assert!(extract_signatures("foo.bar/baz", "anything").is_none());
        // A path with directory dots but a real .rs extension on the
        // final component must still be recognised.
        assert!(
            extract_signatures("foo.bar/baz.rs", "fn main() {}").is_some(),
            "dotted directory must not mask the .rs extension"
        );
        // Absolute paths work too.
        assert!(extract_signatures("/tmp/x.rs", "fn x() {}").is_some());
    }

    #[test]
    fn extract_signatures_detailed_returns_ok_for_known_lang() {
        let result =
            extract_signatures_detailed("lib.rs", "fn x() {}").expect("rust ext supported");
        let sigs = result.expect("extraction must succeed for valid rust input");
        assert!(
            sigs.iter()
                .any(|s| s.kind == "fn" && s.text.contains("fn x"))
        );
    }

    #[test]
    fn extract_signatures_detailed_returns_none_for_unknown_ext() {
        assert!(extract_signatures_detailed("data.json", "{}").is_none());
    }

    #[test]
    fn format_signatures_output() {
        let sigs = vec![
            Signature {
                line: 1,
                kind: "fn",
                text: "pub fn hello(name: &str) -> String".to_owned(),
            },
            Signature {
                line: 5,
                kind: "type",
                text: "pub struct Config".to_owned(),
            },
        ];
        let formatted = format_signatures(&sigs);
        assert!(formatted.contains("L1: [fn] pub fn hello"));
        assert!(formatted.contains("L5: [type] pub struct Config"));
    }

    #[test]
    fn empty_source_returns_empty() {
        let sigs = extract_signatures("lib.rs", "").expect("rust ext supported");
        assert!(sigs.is_empty());
    }

    #[test]
    fn malformed_source_does_not_panic() {
        // tree-sitter is error-tolerant; even broken source must yield
        // at most an empty `Vec`, never a panic.
        let source = "fn broken(\n    let oops = ;";
        let sigs = extract_signatures("lib.rs", source).expect("rust ext supported");
        // We don't assert specific content — only that the call returns.
        let _ = sigs;
    }

    #[test]
    fn signatures_are_sorted_by_line() {
        let source = r"
fn b() {}
fn a() {}
";
        let sigs = extract_signatures("lib.rs", source).expect("rust ext supported");
        let lines: Vec<usize> = sigs.iter().map(|s| s.line).collect();
        let mut sorted = lines.clone();
        sorted.sort_unstable();
        assert_eq!(lines, sorted);
    }
}
