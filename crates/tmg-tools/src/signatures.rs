//! Extract function signatures and type definitions from source files.
//!
//! Provides a lightweight, regex-based extraction of structural definitions
//! (function signatures, struct/class/trait definitions, impl blocks) from
//! source code. This is used by the context compression system to replace
//! full file contents with a structural summary, reducing token usage.
//!
//! Supported languages:
//! - Rust (`.rs`)
//! - Python (`.py`)
//! - JavaScript / TypeScript (`.js`, `.jsx`, `.ts`, `.tsx`)

use std::fmt::Write as _;
use std::sync::LazyLock;

use regex::Regex;

/// A single extracted signature from a source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    /// The line number (1-based) where this signature starts.
    pub line: usize,
    /// The kind of definition (e.g., "fn", "struct", "class").
    pub kind: &'static str,
    /// The extracted signature text.
    pub text: String,
}

/// Detect language from a file extension and extract signatures.
///
/// Returns `None` if the file extension is not recognized as a
/// supported language. Returns an empty `Vec` if no signatures
/// are found.
// TODO: integrate into `ContextCompressor` in tmg-core to replace large
// tool-result file contents with structural summaries during compression.
pub fn extract_signatures(filename: &str, source: &str) -> Option<Vec<Signature>> {
    let ext = filename.rsplit('.').next()?;
    match ext {
        "rs" => Some(extract_rust(source)),
        "py" => Some(extract_python(source)),
        "js" | "jsx" | "ts" | "tsx" | "mjs" | "mts" => Some(extract_javascript(source)),
        _ => None,
    }
}

/// Format extracted signatures into a condensed string summary.
///
/// The output shows line numbers and definition types, suitable for
/// inclusion in compressed context.
pub fn format_signatures(signatures: &[Signature]) -> String {
    let mut buf = String::new();
    for sig in signatures {
        let _ = writeln!(buf, "L{}: [{}] {}", sig.line, sig.kind, sig.text);
    }
    buf
}

// ---------------------------------------------------------------------------
// Rust
// ---------------------------------------------------------------------------

fn extract_rust(source: &str) -> Vec<Signature> {
    // Compile-time-known patterns: compiled once via LazyLock.
    // unwrap is safe here because these are known-valid regex literals.
    #[expect(clippy::unwrap_used, reason = "compile-time-known regex pattern")]
    static FN_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?m)^[[:blank:]]*((?:pub(?:\([^)]*\))?\s+)?(?:(?:async|unsafe|const)\s+)*fn\s+\w+[^{;]*)",
        )
        .unwrap()
    });
    #[expect(clippy::unwrap_used, reason = "compile-time-known regex pattern")]
    static TYPE_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?m)^[[:blank:]]*((?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|type|union)\s+\w+[^{;=]*)",
        )
        .unwrap()
    });
    #[expect(clippy::unwrap_used, reason = "compile-time-known regex pattern")]
    static IMPL_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?m)^[[:blank:]]*(impl(?:<[^>]*>)?\s+(?:\w+(?:::\w+)*(?:<[^>]*>)?\s+for\s+)?\w+(?:::\w+)*(?:<[^>]*>)?)",
        )
        .unwrap()
    });
    #[expect(clippy::unwrap_used, reason = "compile-time-known regex pattern")]
    static MACRO_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?m)^[[:blank:]]*((?:pub(?:\([^)]*\))?\s+)?macro_rules!\s+\w+)").unwrap()
    });

    let mut sigs = Vec::new();
    collect_matches(source, &FN_RE, "fn", &mut sigs);
    collect_matches(source, &TYPE_RE, "type", &mut sigs);
    collect_matches(source, &IMPL_RE, "impl", &mut sigs);
    collect_matches(source, &MACRO_RE, "macro", &mut sigs);

    sigs.sort_by_key(|s| s.line);
    sigs
}

// ---------------------------------------------------------------------------
// Python
// ---------------------------------------------------------------------------

fn extract_python(source: &str) -> Vec<Signature> {
    #[expect(clippy::unwrap_used, reason = "compile-time-known regex pattern")]
    static FN_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?m)^[[:blank:]]*((?:async\s+)?def\s+\w+\s*\([^)]*\)(?:\s*->\s*[^\n:]+)?)")
            .unwrap()
    });
    #[expect(clippy::unwrap_used, reason = "compile-time-known regex pattern")]
    static CLASS_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?m)^[[:blank:]]*(class\s+\w+(?:\([^)]*\))?)").unwrap());

    let mut sigs = Vec::new();
    collect_matches(source, &FN_RE, "def", &mut sigs);
    collect_matches(source, &CLASS_RE, "class", &mut sigs);

    sigs.sort_by_key(|s| s.line);
    sigs
}

// ---------------------------------------------------------------------------
// JavaScript / TypeScript
// ---------------------------------------------------------------------------

fn extract_javascript(source: &str) -> Vec<Signature> {
    #[expect(clippy::unwrap_used, reason = "compile-time-known regex pattern")]
    static FN_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?m)^[[:blank:]]*((?:export\s+)?(?:async\s+)?function\s*\*?\s*\w+\s*(?:<[^>]*>)?\s*\([^)]*\)(?:\s*:\s*[^\n{]+)?)",
        )
        .unwrap()
    });
    #[expect(clippy::unwrap_used, reason = "compile-time-known regex pattern")]
    static ARROW_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?m)^[[:blank:]]*((?:export\s+)?(?:const|let|var)\s+\w+\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?(?:\([^)]*\)|[a-zA-Z_]\w*)(?:\s*:\s*[^\n=>]+)?\s*=>)",
        )
        .unwrap()
    });
    #[expect(clippy::unwrap_used, reason = "compile-time-known regex pattern")]
    static CLASS_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?m)^[[:blank:]]*((?:export\s+)?(?:abstract\s+)?class\s+\w+(?:<[^>]*>)?(?:\s+extends\s+\w+(?:<[^>]*>)?)?(?:\s+implements\s+[^\n{]+)?)",
        )
        .unwrap()
    });
    #[expect(clippy::unwrap_used, reason = "compile-time-known regex pattern")]
    static IFACE_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?m)^[[:blank:]]*((?:export\s+)?interface\s+\w+(?:<[^>]*>)?(?:\s+extends\s+[^\n{]+)?)",
        )
        .unwrap()
    });
    #[expect(clippy::unwrap_used, reason = "compile-time-known regex pattern")]
    static TYPE_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?m)^[[:blank:]]*((?:export\s+)?type\s+\w+(?:<[^>]*>)?\s*=)").unwrap()
    });

    let mut sigs = Vec::new();
    collect_matches(source, &FN_RE, "function", &mut sigs);
    collect_matches(source, &ARROW_RE, "const fn", &mut sigs);
    collect_matches(source, &CLASS_RE, "class", &mut sigs);
    collect_matches(source, &IFACE_RE, "interface", &mut sigs);
    collect_matches(source, &TYPE_RE, "type", &mut sigs);

    sigs.sort_by_key(|s| s.line);
    sigs
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Collect regex matches from source, computing line numbers.
fn collect_matches(source: &str, re: &Regex, kind: &'static str, sigs: &mut Vec<Signature>) {
    for m in re.find_iter(source) {
        let start = m.start();
        let line = source[..start].matches('\n').count() + 1;
        let text = m.as_str().trim().to_owned();
        // Skip very short matches that are likely false positives.
        if text.len() > 3 {
            sigs.push(Signature { line, kind, text });
        }
    }
}

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
        let sigs = extract_rust(source);
        assert!(sigs.len() >= 3, "expected >= 3 sigs, got {}", sigs.len());
        assert!(sigs.iter().any(|s| s.text.contains("pub fn hello")));
        assert!(sigs.iter().any(|s| s.text.contains("async fn fetch_data")));
        assert!(sigs.iter().any(|s| s.text.contains("unsafe fn raw_ptr")));
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
        let sigs = extract_rust(source);
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
        let sigs = extract_python(source);
        assert!(
            sigs.iter()
                .any(|s| s.kind == "class" && s.text.contains("class MyClass"))
        );
        assert!(
            sigs.iter()
                .any(|s| s.kind == "def" && s.text.contains("def __init__"))
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
    fn javascript_signatures() {
        let source = r"
export function greet(name: string): string {
    return `Hello, ${name}`;
}

export async function fetchData(url: string): Promise<Data> {
    return await fetch(url);
}

const add = (a: number, b: number): number => a + b;

export class UserService extends BaseService implements Serializable {
    constructor(private db: Database) {}
}

interface Config {
    host: string;
    port: number;
}

export type Result<T> = { ok: true; value: T } | { ok: false; error: Error };
";
        let sigs = extract_javascript(source);
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
    }

    #[test]
    fn detect_language_from_extension() {
        assert!(extract_signatures("main.rs", "fn main() {}").is_some());
        assert!(extract_signatures("app.py", "def main(): pass").is_some());
        assert!(extract_signatures("index.ts", "function main() {}").is_some());
        assert!(extract_signatures("app.tsx", "function App() {}").is_some());
        assert!(extract_signatures("data.json", "{}").is_none());
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
        let sigs = extract_rust("");
        assert!(sigs.is_empty());
    }
}
