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

use std::fmt::Write as _;
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
/// language. Returns an empty `Vec` when the file parses cleanly but
/// contains no top-level definitions (e.g. an empty source).
///
/// On parse failure (out-of-range tree-sitter ABI, allocation failure,
/// or a panic-free internal error), this returns `Some(vec![])` so the
/// caller can distinguish "language not supported" from "no signatures
/// found".
#[must_use]
pub fn extract_signatures(filename: &str, source: &str) -> Option<Vec<Signature>> {
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
        let ext = filename.rsplit('.').next()?;
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
            "struct" | "enum" | "trait" | "type_alias" | "union" | "ts_type" => Some("type"),
            "impl" => Some("impl"),
            "macro" => Some("macro"),
            "py_fn" => Some("def"),
            "py_class" | "class" => Some("class"),
            "function" => Some("function"),
            "interface" => Some("interface"),
            _ => None,
        }
    }

    fn extract(self, source: &str) -> Vec<Signature> {
        let mut parser = Parser::new();
        if parser.set_language(&self.ts_language()).is_err() {
            return Vec::new();
        }
        let Some(tree) = parser.parse(source, None) else {
            return Vec::new();
        };
        let query = match self {
            Self::Rust => &*RUST_QUERY_COMPILED,
            Self::Python => &*PYTHON_QUERY_COMPILED,
            Self::TypeScript => &*TS_QUERY_COMPILED,
            Self::Tsx => &*TSX_QUERY_COMPILED,
            Self::JavaScript => &*JS_QUERY_COMPILED,
        };

        // Empty queries (compile failure) yield no captures, so fall back
        // gracefully without panicking.
        if query.pattern_count() == 0 {
            return Vec::new();
        }

        collect_from_query(query, &tree.root_node(), source.as_bytes())
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

const TS_QUERY: &str = r"
(function_declaration) @function
(class_declaration) @class
(interface_declaration) @interface
(type_alias_declaration) @ts_type
";

const JS_QUERY: &str = r"
(function_declaration) @function
(class_declaration) @class
";

// Compile each query lazily once so repeated extract calls don't pay
// the parser-construction cost. We use `LazyLock<Query>` because every
// language exposes a `LANGUAGE` const; a compile failure here would
// indicate a tree-sitter ABI mismatch (in which case we want to know
// loudly via tests rather than silently fall back).
static RUST_QUERY_COMPILED: LazyLock<Query> = LazyLock::new(|| {
    Query::new(&Lang::Rust.ts_language(), RUST_QUERY).unwrap_or_else(|_| empty_query())
});
static PYTHON_QUERY_COMPILED: LazyLock<Query> = LazyLock::new(|| {
    Query::new(&Lang::Python.ts_language(), PYTHON_QUERY).unwrap_or_else(|_| empty_query())
});
static TS_QUERY_COMPILED: LazyLock<Query> = LazyLock::new(|| {
    Query::new(&Lang::TypeScript.ts_language(), TS_QUERY).unwrap_or_else(|_| empty_query())
});
static TSX_QUERY_COMPILED: LazyLock<Query> = LazyLock::new(|| {
    Query::new(&Lang::Tsx.ts_language(), TS_QUERY).unwrap_or_else(|_| empty_query())
});
static JS_QUERY_COMPILED: LazyLock<Query> = LazyLock::new(|| {
    Query::new(&Lang::JavaScript.ts_language(), JS_QUERY).unwrap_or_else(|_| empty_query())
});

/// Build an empty `Query` as a fallback when compilation fails. We only
/// reach this if a language ABI mismatches at runtime; in that case the
/// extractor returns an empty `Vec` and the caller falls back to tail
/// truncation.
fn empty_query() -> Query {
    // An empty pattern set produces zero matches but is constructible
    // for any language. We use Rust here because it always loads.
    Query::new(&Lang::Rust.ts_language(), "").unwrap_or_else(|_| {
        // Truly should never happen: fail closed by panicking only if
        // the most basic empty query cannot compile, which would mean
        // tree-sitter itself is broken.
        unreachable!("tree-sitter cannot construct an empty query");
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
