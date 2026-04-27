//! `${{ ... }}` expression language (SPEC §8.6).
//!
//! A small recursive-descent evaluator. The grammar (in pseudo-EBNF):
//!
//! ```text
//! expr        := or_expr
//! or_expr     := and_expr ( "||" and_expr )*
//! and_expr    := comparison ( "&&" comparison )*
//! comparison  := unary ( ("==" | "!=" | "<" | ">" | "<=" | ">=") unary )?
//! unary       := "!" unary | primary
//! primary     := literal | path | "(" expr ")"
//! literal     := integer | string | bool | "null"
//! path        := identifier ( "." identifier | "[" key "]" )*
//! key         := string | identifier
//! ```
//!
//! `!` is a unary prefix that binds tighter than `==`/`!=`/etc., so
//! `!a == b` parses as `(!a) == b`. Use `!(a == b)` to negate the
//! whole comparison.
//!
//! Three top-level entry points:
//!
//! - [`eval_string`] — substitutes every `${{ ... }}` occurrence in a
//!   template string with the value's display rendering.
//! - [`eval_bool`] — parses a single expression and coerces it to bool.
//! - [`eval_value`] — parses a single expression and returns the raw
//!   `serde_json::Value`.
//!
//! ## Type coercion
//!
//! Intentionally conservative; documented here so workflow authors get
//! deterministic behaviour:
//!
//! - **Equality (`==` / `!=`)** compares JSON values structurally. There
//!   is no implicit string-to-int coercion: `1 == "1"` is `false`.
//! - **Order comparisons (`<`, `<=`, `>`, `>=`)** require both operands
//!   to be numbers; comparing a string with a number raises an error.
//! - **Truthiness** (used by `!`, `&&`, `||`, and `eval_bool`) follows
//!   JSON conventions: `false`, `null`, `0`, `""`, `[]`, `{}` are
//!   falsy; everything else is truthy.
//! - **`!` precedence**: `!` binds tighter than comparison, so
//!   `!a == b` parses as `(!a) == b`. Use `!(a == b)` for the
//!   comparison-then-negate form.
//! - **`&&` / `||` always return `Value::Bool`** — this differs from
//!   JS/Python, which return the chosen *operand* value. We coerce the
//!   result to `Bool(truthy(...))` for both operators, so chaining
//!   like `${{ inputs.name || "default" }}` does **not** fall through
//!   to the right-hand string. Use a `${{ when: ... }}` or an explicit
//!   conditional in the workflow instead.
//!
//!   Example: `"x" || "y"` evaluates to `Bool(true)`, not the string
//!   `"y"`; `1 && 2` evaluates to `Bool(true)`, not `Bool(false)`
//!   (both operands are truthy) and not `2`.
//! - **Identifier chains** return a clear error message for missing
//!   scope keys ("unknown identifier 'foo' in 'inputs'").

use std::collections::BTreeMap;

use serde_json::Value;

use crate::def::StepResult;
use crate::error::{Result, WorkflowError};

/// Compute a 1-based `(line, col)` for a byte offset into `src`.
///
/// Lines are split on `\n`; columns are 1-based char counts within the
/// current line. The returned `col` is char-based, not byte-based, so
/// multi-byte characters do not skew the column. Used by error
/// reporters in this module to emit `at line {l}, col {c} (byte {p})`
/// messages alongside the byte offset.
fn line_col(src: &str, pos: usize) -> (usize, usize) {
    // Clamp pos so a buggy caller can't panic the slicing below.
    let pos = pos.min(src.len());
    let prefix = &src[..pos];
    let mut line = 1usize;
    let mut last_newline = 0usize;
    for (idx, b) in prefix.bytes().enumerate() {
        if b == b'\n' {
            line += 1;
            last_newline = idx + 1;
        }
    }
    let col = src[last_newline..pos].chars().count() + 1;
    (line, col)
}

/// The four scopes available to expressions.
pub struct ExprContext<'a> {
    /// Workflow inputs as a JSON object.
    pub inputs: &'a Value,
    /// Per-step results keyed by step id.
    pub steps: &'a BTreeMap<String, StepResult>,
    /// `tsumugi.toml`-derived configuration as a JSON object.
    pub config: &'a Value,
    /// Environment variables.
    pub env: &'a BTreeMap<String, String>,
}

impl<'a> ExprContext<'a> {
    /// Construct a new context.
    #[must_use]
    pub fn new(
        inputs: &'a Value,
        steps: &'a BTreeMap<String, StepResult>,
        config: &'a Value,
        env: &'a BTreeMap<String, String>,
    ) -> Self {
        Self {
            inputs,
            steps,
            config,
            env,
        }
    }
}

/// Substitute every `${{ ... }}` template in `template` with the
/// rendered value of the expression inside.
pub fn eval_string(template: &str, ctx: &ExprContext) -> Result<String> {
    let mut out = String::with_capacity(template.len());
    let bytes = template.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Look for the start sequence `${{`. Note the literal-`$`
        // escape `$${{` is *not* supported in this iteration; YAML
        // already gives users dollar-safe quoting and the SPEC has no
        // escape syntax. Document any future escape needs in #41.
        if i + 2 < bytes.len() && &bytes[i..i + 3] == b"${{" {
            // Find the matching `}}`.
            let start = i + 3;
            let Some(end_offset) = find_close(&bytes[start..]) else {
                let (line, col) = line_col(template, i);
                return Err(WorkflowError::expression(format!(
                    "unclosed '${{{{' in template starting at line {line}, col {col} (byte {i})"
                )));
            };
            let end = start + end_offset;
            let expr_src = std::str::from_utf8(&bytes[start..end]).map_err(|_| {
                WorkflowError::expression("non-UTF-8 bytes inside ${{ ... }} expression")
            })?;
            let value = eval_value(expr_src.trim(), ctx)?;
            out.push_str(&render_value(&value));
            i = end + 2; // skip past `}}`
        } else {
            // Append a single character. We branch on raw ASCII bytes
            // (`${{`) for speed, then fall through to UTF-8 char
            // decoding here so that multi-byte characters are appended
            // atomically. Using `chars().next()` keeps `out` valid
            // UTF-8 without needing `unsafe` byte-level pushing.
            if let Some(ch) = template[i..].chars().next() {
                out.push(ch);
                i += ch.len_utf8();
            } else {
                break;
            }
        }
    }
    Ok(out)
}

/// Find the byte offset of the matching `}}` close in `tail`, ignoring
/// `}}` inside string literals.
///
/// Inside a string literal a backslash (`\\`) escapes the next byte —
/// in particular `\"` does *not* close a `"`-delimited string, and
/// `\'` does not close a `'`-delimited one. Without this, the
/// expression `${{ "a\"b" }}` would be reported as unclosed because
/// the scanner would see the embedded `\"` as a string close, treat
/// the rest as out-of-string code, and never find a `}}` while still
/// inside the (re-entered) string.
fn find_close(tail: &[u8]) -> Option<usize> {
    let mut i = 0;
    let mut in_string: Option<u8> = None; // delimiter byte if inside a string
    while i + 1 < tail.len() {
        let b = tail[i];
        match in_string {
            Some(delim) => {
                if b == b'\\' {
                    // Skip the escape byte plus the next byte. If the
                    // backslash is the last byte, advance once and let
                    // the loop terminate naturally (the string is
                    // unterminated; the parser will report it later).
                    i += 2;
                } else if b == delim {
                    in_string = None;
                    i += 1;
                } else {
                    i += 1;
                }
            }
            None => {
                if b == b'"' || b == b'\'' {
                    in_string = Some(b);
                    i += 1;
                } else if b == b'}' && tail[i + 1] == b'}' {
                    return Some(i);
                } else {
                    i += 1;
                }
            }
        }
    }
    None
}

/// Render a JSON value as a human-readable string for template
/// substitution. Strings are emitted bare (no surrounding quotes);
/// scalars stringify naturally; arrays and objects fall back to JSON.
fn render_value(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

/// Evaluate an expression and coerce to bool.
///
/// The input is the *expression body* (no surrounding `${{ ... }}`).
pub fn eval_bool(expr: &str, ctx: &ExprContext) -> Result<bool> {
    let value = eval_value(expr, ctx)?;
    Ok(truthy(&value))
}

/// Evaluate an expression and return the raw JSON value.
pub fn eval_value(expr: &str, ctx: &ExprContext) -> Result<Value> {
    let mut parser = Parser::new(expr);
    let value = parser.parse_or()?;
    parser.skip_ws();
    if !parser.is_eof() {
        let (line, col) = line_col(expr, parser.pos);
        return Err(WorkflowError::expression(format!(
            "unexpected trailing characters at line {line}, col {col} (byte {pos}): '{tail}'",
            pos = parser.pos,
            tail = parser.tail(),
        )));
    }
    evaluate(value, ctx)
}

/// JSON-style truthiness used by `!`, `&&`, `||`, `when`.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

// ---------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------

/// AST nodes produced by the parser.
#[derive(Debug, Clone)]
enum Node {
    Literal(Value),
    Path(Vec<PathSeg>),
    Not(Box<Node>),
    BinOp(Op, Box<Node>, Box<Node>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    And,
    Or,
}

#[derive(Debug, Clone)]
enum PathSeg {
    Ident(String),
    Index(String),
}

struct Parser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn tail(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
    }

    /// Try to consume a literal token; returns `true` if matched.
    fn eat(&mut self, lit: &str) -> bool {
        if self.tail().starts_with(lit) {
            self.pos += lit.len();
            true
        } else {
            false
        }
    }

    /// `or_expr := and_expr ( "||" and_expr )*`
    fn parse_or(&mut self) -> Result<Node> {
        let mut left = self.parse_and()?;
        loop {
            self.skip_ws();
            if self.eat("||") {
                let right = self.parse_and()?;
                left = Node::BinOp(Op::Or, Box::new(left), Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }

    /// `and_expr := comparison ( "&&" comparison )*`
    fn parse_and(&mut self) -> Result<Node> {
        let mut left = self.parse_comparison()?;
        loop {
            self.skip_ws();
            if self.eat("&&") {
                let right = self.parse_comparison()?;
                left = Node::BinOp(Op::And, Box::new(left), Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }

    /// `unary := "!" unary | primary`
    ///
    /// `!` binds tighter than comparison, so `!a == b` parses as
    /// `(!a) == b`. This is the canonical unary-vs-binary precedence:
    /// the unary prefix is consumed by `parse_unary`, and the binary
    /// comparison operators are consumed at the next level up
    /// (`parse_comparison`), which calls `parse_unary` for both
    /// operands. Use `!(a == b)` explicitly to negate the whole
    /// comparison.
    ///
    /// Stacked unary (`!!a`) is supported via the recursive call.
    fn parse_unary(&mut self) -> Result<Node> {
        self.skip_ws();
        // Be careful: `!=` starts with `!` too.
        if self.tail().starts_with('!') && !self.tail().starts_with("!=") {
            self.pos += 1;
            let inner = self.parse_unary()?;
            return Ok(Node::Not(Box::new(inner)));
        }
        self.parse_primary()
    }

    /// `comparison := unary ( ( "==" | "!=" | "<=" | ">=" | "<" | ">" ) unary )?`
    fn parse_comparison(&mut self) -> Result<Node> {
        let left = self.parse_unary()?;
        self.skip_ws();
        let op = if self.eat("==") {
            Op::Eq
        } else if self.eat("!=") {
            Op::Neq
        } else if self.eat("<=") {
            Op::Lte
        } else if self.eat(">=") {
            Op::Gte
        } else if self.eat("<") {
            Op::Lt
        } else if self.eat(">") {
            Op::Gt
        } else {
            return Ok(left);
        };
        let right = self.parse_unary()?;
        Ok(Node::BinOp(op, Box::new(left), Box::new(right)))
    }

    /// `primary := literal | path | "(" expr ")"`
    fn parse_primary(&mut self) -> Result<Node> {
        self.skip_ws();
        let Some(c) = self.peek() else {
            return Err(WorkflowError::expression("unexpected end of expression"));
        };

        if c == '(' {
            self.pos += 1;
            let inner = self.parse_or()?;
            self.skip_ws();
            if !self.eat(")") {
                return Err(WorkflowError::expression(
                    "expected closing ')' in expression",
                ));
            }
            return Ok(inner);
        }

        if c == '"' || c == '\'' {
            let s = self.parse_string_literal()?;
            return Ok(Node::Literal(Value::String(s)));
        }

        if c.is_ascii_digit() || c == '-' {
            return self.parse_number();
        }

        if c.is_ascii_alphabetic() || c == '_' {
            // bool / null literal or path expression.
            let ident = self.read_identifier()?;
            match ident.as_str() {
                "true" => return Ok(Node::Literal(Value::Bool(true))),
                "false" => return Ok(Node::Literal(Value::Bool(false))),
                "null" => return Ok(Node::Literal(Value::Null)),
                _ => {}
            }
            let mut segs = vec![PathSeg::Ident(ident)];
            self.continue_path(&mut segs)?;
            return Ok(Node::Path(segs));
        }

        let (line, col) = line_col(self.src, self.pos);
        Err(WorkflowError::expression(format!(
            "unexpected character '{c}' at line {line}, col {col} (byte {pos})",
            pos = self.pos,
        )))
    }

    fn parse_number(&mut self) -> Result<Node> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.pos += 1;
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        // Optional fractional / exponent part.
        if self.peek() == Some('.') {
            self.pos += 1;
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }
        let raw = &self.src[start..self.pos];
        if raw == "-" {
            let (line, col) = line_col(self.src, start);
            return Err(WorkflowError::expression(format!(
                "expected number after '-' at line {line}, col {col} (byte {start})"
            )));
        }
        // Try integer first, then float.
        if let Ok(n) = raw.parse::<i64>() {
            return Ok(Node::Literal(Value::Number(n.into())));
        }
        if let Ok(f) = raw.parse::<f64>() {
            if let Some(num) = serde_json::Number::from_f64(f) {
                return Ok(Node::Literal(Value::Number(num)));
            }
        }
        Err(WorkflowError::expression(format!(
            "invalid numeric literal '{raw}'"
        )))
    }

    fn parse_string_literal(&mut self) -> Result<String> {
        let Some(delim) = self.bump() else {
            return Err(WorkflowError::expression("expected string literal"));
        };
        let mut out = String::new();
        loop {
            match self.bump() {
                None => {
                    return Err(WorkflowError::expression("unterminated string literal"));
                }
                Some('\\') => {
                    let Some(esc) = self.bump() else {
                        return Err(WorkflowError::expression("trailing backslash"));
                    };
                    match esc {
                        'n' => out.push('\n'),
                        't' => out.push('\t'),
                        'r' => out.push('\r'),
                        '\\' => out.push('\\'),
                        '"' => out.push('"'),
                        '\'' => out.push('\''),
                        other => out.push(other),
                    }
                }
                Some(c) if c == delim => return Ok(out),
                Some(c) => out.push(c),
            }
        }
    }

    fn read_identifier(&mut self) -> Result<String> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
        if start == self.pos {
            return Err(WorkflowError::expression("expected identifier"));
        }
        Ok(self.src[start..self.pos].to_owned())
    }

    fn continue_path(&mut self, segs: &mut Vec<PathSeg>) -> Result<()> {
        loop {
            self.skip_ws();
            if self.eat(".") {
                let ident = self.read_identifier()?;
                segs.push(PathSeg::Ident(ident));
            } else if self.eat("[") {
                self.skip_ws();
                let key = if matches!(self.peek(), Some('"' | '\'')) {
                    self.parse_string_literal()?
                } else if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    // Numeric array index.
                    let start = self.pos;
                    while let Some(c) = self.peek() {
                        if c.is_ascii_digit() {
                            self.pos += 1;
                        } else {
                            break;
                        }
                    }
                    self.src[start..self.pos].to_owned()
                } else {
                    self.read_identifier()?
                };
                self.skip_ws();
                if !self.eat("]") {
                    return Err(WorkflowError::expression(
                        "expected ']' in indexing expression",
                    ));
                }
                segs.push(PathSeg::Index(key));
            } else {
                return Ok(());
            }
        }
    }
}

/// Evaluate an AST [`Node`] against the given context.
///
/// Free function rather than `impl` method because the parser carries
/// no state once a `Node` has been built; making `evaluate` a method
/// triggered clippy's `self_only_used_in_recursion` lint.
fn evaluate(node: Node, ctx: &ExprContext) -> Result<Value> {
    Ok(match node {
        Node::Literal(v) => v,
        Node::Path(segs) => resolve_path(&segs, ctx)?,
        Node::Not(inner) => Value::Bool(!truthy(&evaluate(*inner, ctx)?)),
        Node::BinOp(op, l, r) => match op {
            Op::And => {
                let lv = evaluate(*l, ctx)?;
                if truthy(&lv) {
                    Value::Bool(truthy(&evaluate(*r, ctx)?))
                } else {
                    Value::Bool(false)
                }
            }
            Op::Or => {
                let lv = evaluate(*l, ctx)?;
                if truthy(&lv) {
                    Value::Bool(true)
                } else {
                    Value::Bool(truthy(&evaluate(*r, ctx)?))
                }
            }
            Op::Eq => Value::Bool(json_eq(&evaluate(*l, ctx)?, &evaluate(*r, ctx)?)),
            Op::Neq => Value::Bool(!json_eq(&evaluate(*l, ctx)?, &evaluate(*r, ctx)?)),
            Op::Lt | Op::Gt | Op::Lte | Op::Gte => {
                let lv = evaluate(*l, ctx)?;
                let rv = evaluate(*r, ctx)?;
                Value::Bool(compare_ord(op, &lv, &rv)?)
            }
        },
    })
}

fn json_eq(a: &Value, b: &Value) -> bool {
    // No implicit string<->number coercion — `1 == "1"` is `false` per
    // the documented contract.
    a == b
}

fn compare_ord(op: Op, a: &Value, b: &Value) -> Result<bool> {
    let af = a.as_f64().ok_or_else(|| {
        WorkflowError::expression(format!(
            "left operand of order comparison must be numeric, got {}",
            value_type(a)
        ))
    })?;
    let bf = b.as_f64().ok_or_else(|| {
        WorkflowError::expression(format!(
            "right operand of order comparison must be numeric, got {}",
            value_type(b)
        ))
    })?;
    Ok(match op {
        Op::Lt => af < bf,
        Op::Gt => af > bf,
        Op::Lte => af <= bf,
        Op::Gte => af >= bf,
        _ => false,
    })
}

fn value_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn resolve_path(segs: &[PathSeg], ctx: &ExprContext) -> Result<Value> {
    let Some(first) = segs.first() else {
        return Err(WorkflowError::expression("empty identifier chain"));
    };
    let PathSeg::Ident(scope) = first else {
        return Err(WorkflowError::expression(
            "expression must start with a scope identifier",
        ));
    };

    // Resolve the scope head into a `Value` (potentially synthesized
    // from `StepResult`).
    let head = match scope.as_str() {
        "inputs" => ctx.inputs.clone(),
        "config" => ctx.config.clone(),
        "env" => {
            let map: serde_json::Map<String, Value> = ctx
                .env
                .iter()
                .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                .collect();
            Value::Object(map)
        }
        "steps" => {
            // We materialize steps lazily: only follow into the step
            // we need. The next segment must be the step id.
            return resolve_steps_path(&segs[1..], ctx);
        }
        other => {
            return Err(WorkflowError::expression(format!(
                "unknown top-level identifier '{other}' (allowed: inputs, steps, config, env)"
            )));
        }
    };

    walk_path(&head, &segs[1..], scope)
}

fn resolve_steps_path(rest: &[PathSeg], ctx: &ExprContext) -> Result<Value> {
    let Some(first) = rest.first() else {
        // Bare `steps` reference — return empty object snapshot.
        return Ok(Value::Object(serde_json::Map::new()));
    };
    let step_id = match first {
        PathSeg::Ident(s) | PathSeg::Index(s) => s,
    };

    let Some(result) = ctx.steps.get(step_id) else {
        return Err(WorkflowError::expression(format!(
            "unknown step '{step_id}' in 'steps' (no such id has run yet)"
        )));
    };

    // Materialize the StepResult into a JSON object. This snapshot is
    // cheap and lets the same `walk_path` machinery handle the
    // remaining segments.
    let mut obj = serde_json::Map::new();
    obj.insert("output".to_owned(), result.output.clone());
    obj.insert(
        "exit_code".to_owned(),
        Value::Number(serde_json::Number::from(i64::from(result.exit_code))),
    );
    obj.insert("stdout".to_owned(), Value::String(result.stdout.clone()));
    obj.insert("stderr".to_owned(), Value::String(result.stderr.clone()));
    obj.insert(
        "changed_files".to_owned(),
        Value::Array(
            result
                .changed_files
                .iter()
                .map(|s| Value::String(s.clone()))
                .collect(),
        ),
    );
    let snapshot = Value::Object(obj);

    walk_path(&snapshot, &rest[1..], &format!("steps.{step_id}"))
}

fn walk_path(start: &Value, segs: &[PathSeg], base_label: &str) -> Result<Value> {
    let mut current = start.clone();
    let mut current_label = base_label.to_owned();
    for seg in segs {
        match seg {
            PathSeg::Ident(name) | PathSeg::Index(name) => {
                current = match &current {
                    Value::Object(map) => map.get(name).cloned().ok_or_else(|| {
                        WorkflowError::expression(format!(
                            "unknown identifier '{name}' in '{current_label}'"
                        ))
                    })?,
                    Value::Array(arr) => {
                        let idx: usize = name.parse().map_err(|_| {
                            WorkflowError::expression(format!(
                                "non-numeric index '{name}' on array '{current_label}'"
                            ))
                        })?;
                        arr.get(idx).cloned().ok_or_else(|| {
                            WorkflowError::expression(format!(
                                "index {idx} out of bounds for array '{current_label}' (len {})",
                                arr.len()
                            ))
                        })?
                    }
                    other => {
                        return Err(WorkflowError::expression(format!(
                            "cannot index into {} value at '{current_label}'",
                            value_type(other)
                        )));
                    }
                };
                current_label = format!("{current_label}.{name}");
            }
        }
    }
    Ok(current)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions")]
mod tests {
    use super::*;
    use serde_json::json;

    fn ctx_with_inputs(
        inputs: Value,
    ) -> (
        Value,
        BTreeMap<String, StepResult>,
        Value,
        BTreeMap<String, String>,
    ) {
        (
            inputs,
            BTreeMap::new(),
            Value::Object(serde_json::Map::new()),
            BTreeMap::new(),
        )
    }

    fn make_ctx<'a>(
        inputs: &'a Value,
        steps: &'a BTreeMap<String, StepResult>,
        config: &'a Value,
        env: &'a BTreeMap<String, String>,
    ) -> ExprContext<'a> {
        ExprContext::new(inputs, steps, config, env)
    }

    #[test]
    fn literals() {
        let (i, s, c, e) = ctx_with_inputs(Value::Null);
        let ctx = make_ctx(&i, &s, &c, &e);
        assert_eq!(eval_value("42", &ctx).unwrap(), json!(42));
        assert_eq!(eval_value("\"hi\"", &ctx).unwrap(), json!("hi"));
        assert_eq!(eval_value("'single'", &ctx).unwrap(), json!("single"));
        assert_eq!(eval_value("true", &ctx).unwrap(), json!(true));
        assert_eq!(eval_value("false", &ctx).unwrap(), json!(false));
        assert_eq!(eval_value("null", &ctx).unwrap(), json!(null));
        assert_eq!(eval_value("-7", &ctx).unwrap(), json!(-7));
    }

    #[test]
    fn comparisons() {
        let (i, s, c, e) = ctx_with_inputs(Value::Null);
        let ctx = make_ctx(&i, &s, &c, &e);
        assert!(eval_bool("1 == 1", &ctx).unwrap());
        assert!(!eval_bool("1 == 2", &ctx).unwrap());
        // No string<->int coercion.
        assert!(!eval_bool("1 == \"1\"", &ctx).unwrap());
        assert!(eval_bool("\"a\" != \"b\"", &ctx).unwrap());
        assert!(eval_bool("3 < 5", &ctx).unwrap());
        assert!(eval_bool("5 >= 5", &ctx).unwrap());
        assert!(!eval_bool("5 > 5", &ctx).unwrap());
    }

    #[test]
    fn order_comparison_requires_number() {
        let (i, s, c, e) = ctx_with_inputs(Value::Null);
        let ctx = make_ctx(&i, &s, &c, &e);
        let err = eval_bool("\"a\" < 1", &ctx).unwrap_err();
        assert!(err.to_string().contains("must be numeric"), "{err}");
    }

    #[test]
    fn logical_short_circuit_and() {
        // `false && undefined` must not raise: the `&&` short-circuits.
        let (i, s, c, e) = ctx_with_inputs(json!({"x": 1}));
        let ctx = make_ctx(&i, &s, &c, &e);
        assert!(!eval_bool("false && inputs.missing", &ctx).unwrap());
    }

    #[test]
    fn logical_short_circuit_or() {
        let (i, s, c, e) = ctx_with_inputs(json!({"x": 1}));
        let ctx = make_ctx(&i, &s, &c, &e);
        assert!(eval_bool("true || inputs.missing", &ctx).unwrap());
    }

    #[test]
    fn precedence_and_over_or() {
        // `false || true && false` should be `false || (true && false)` = false.
        let (i, s, c, e) = ctx_with_inputs(Value::Null);
        let ctx = make_ctx(&i, &s, &c, &e);
        assert!(!eval_bool("false || true && false", &ctx).unwrap());
        assert!(eval_bool("(false || true) && true", &ctx).unwrap());
    }

    #[test]
    fn negation() {
        let (i, s, c, e) = ctx_with_inputs(Value::Null);
        let ctx = make_ctx(&i, &s, &c, &e);
        assert!(eval_bool("!false", &ctx).unwrap());
        assert!(!eval_bool("!true", &ctx).unwrap());
        assert!(!eval_bool("!1", &ctx).unwrap());
    }

    /// `!` binds tighter than comparison: `!1 == 0` parses as
    /// `(!1) == 0`. With no implicit number/bool coercion, `!1` is
    /// `false` and `false == 0` is `false`.
    #[test]
    fn not_binds_tighter_than_comparison_literal_form() {
        let (i, s, c, e) = ctx_with_inputs(Value::Null);
        let ctx = make_ctx(&i, &s, &c, &e);
        // (!1) == 0 -> false == 0 -> false (no coercion)
        assert!(!eval_bool("!1 == 0", &ctx).unwrap());
    }

    /// `!a == b` parses as `(!a) == b`. With `a=true, b=false`:
    /// `(!true) == false` -> `false == false` -> `true`.
    #[test]
    fn not_binds_tighter_than_comparison_via_inputs() {
        let inputs = json!({"a": true, "b": false});
        let s = BTreeMap::new();
        let c = Value::Null;
        let e = BTreeMap::new();
        let ctx = make_ctx(&inputs, &s, &c, &e);
        assert!(eval_bool("!inputs.a == inputs.b", &ctx).unwrap());
    }

    /// Distinguishing case where the two parses would actually
    /// disagree. With `a=0, b=false`:
    ///   - correct `(!a) == b` -> `(!0) == false` -> `true == false`
    ///     -> `false` (no Number/Bool coercion).
    ///   - buggy   `!(a == b)` -> `!(0 == false)` -> `!false` -> `true`.
    #[test]
    fn not_precedence_disambiguating_case() {
        let inputs = json!({"a": 0, "b": false});
        let s = BTreeMap::new();
        let c = Value::Null;
        let e = BTreeMap::new();
        let ctx = make_ctx(&inputs, &s, &c, &e);
        assert!(!eval_bool("!inputs.a == inputs.b", &ctx).unwrap());
    }

    /// Explicit `!(a == b)` still works: parens override the unary
    /// prefix and let `!` apply to the whole comparison's truthiness.
    #[test]
    fn not_with_parens_negates_whole_comparison() {
        let inputs = json!({"a": true, "b": false});
        let s = BTreeMap::new();
        let c = Value::Null;
        let e = BTreeMap::new();
        let ctx = make_ctx(&inputs, &s, &c, &e);
        // !(a == b) -> !(true == false) -> !false -> true
        assert!(eval_bool("!(inputs.a == inputs.b)", &ctx).unwrap());
    }

    #[test]
    fn inputs_path_resolution() {
        let inputs = json!({
            "name": "alice",
            "deep": {"x": [10, 20, 30]}
        });
        let s = BTreeMap::new();
        let c = Value::Null;
        let e = BTreeMap::new();
        let ctx = make_ctx(&inputs, &s, &c, &e);
        assert_eq!(eval_value("inputs.name", &ctx).unwrap(), json!("alice"));
        assert_eq!(
            eval_value("inputs.deep.x", &ctx).unwrap(),
            json!([10, 20, 30])
        );
        assert_eq!(eval_value("inputs.deep.x[1]", &ctx).unwrap(), json!(20));
        assert_eq!(
            eval_value("inputs[\"name\"]", &ctx).unwrap(),
            json!("alice")
        );
    }

    #[test]
    fn unknown_identifier_clear_error() {
        let inputs = json!({"a": 1});
        let s = BTreeMap::new();
        let c = Value::Null;
        let e = BTreeMap::new();
        let ctx = make_ctx(&inputs, &s, &c, &e);
        let err = eval_value("inputs.b", &ctx).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown identifier 'b'"), "{msg}");
        assert!(msg.contains("inputs"), "{msg}");
    }

    #[test]
    fn unknown_top_level_identifier() {
        let (i, s, c, e) = ctx_with_inputs(Value::Null);
        let ctx = make_ctx(&i, &s, &c, &e);
        let err = eval_value("foo.bar", &ctx).unwrap_err();
        assert!(err.to_string().contains("unknown top-level identifier"));
    }

    #[test]
    fn step_path_resolution() {
        let inputs = Value::Null;
        let mut steps = BTreeMap::new();
        steps.insert(
            "verify".to_owned(),
            StepResult {
                output: json!({"ok": true}),
                exit_code: 0,
                stdout: "OK\n".to_owned(),
                stderr: String::new(),
                changed_files: vec!["a.rs".to_owned()],
            },
        );
        let c = Value::Null;
        let e = BTreeMap::new();
        let ctx = make_ctx(&inputs, &steps, &c, &e);
        assert_eq!(
            eval_value("steps.verify.exit_code", &ctx).unwrap(),
            json!(0),
        );
        assert_eq!(
            eval_value("steps.verify.output.ok", &ctx).unwrap(),
            json!(true),
        );
        assert_eq!(
            eval_value("steps.verify.changed_files[0]", &ctx).unwrap(),
            json!("a.rs"),
        );
    }

    #[test]
    fn unknown_step_id_clear_error() {
        let inputs = Value::Null;
        let s = BTreeMap::new();
        let c = Value::Null;
        let e = BTreeMap::new();
        let ctx = make_ctx(&inputs, &s, &c, &e);
        let err = eval_value("steps.does_not_exist.exit_code", &ctx).unwrap_err();
        assert!(err.to_string().contains("unknown step 'does_not_exist'"));
    }

    #[test]
    fn env_resolution() {
        let inputs = Value::Null;
        let s = BTreeMap::new();
        let c = Value::Null;
        let mut env = BTreeMap::new();
        env.insert("HOME".to_owned(), "/home/me".to_owned());
        let ctx = make_ctx(&inputs, &s, &c, &env);
        assert_eq!(eval_value("env.HOME", &ctx).unwrap(), json!("/home/me"));
    }

    #[test]
    fn template_substitution() {
        let inputs = json!({"name": "world", "n": 3});
        let s = BTreeMap::new();
        let c = Value::Null;
        let e = BTreeMap::new();
        let ctx = make_ctx(&inputs, &s, &c, &e);
        let out = eval_string("Hello, ${{ inputs.name }}! n=${{ inputs.n }}", &ctx).unwrap();
        assert_eq!(out, "Hello, world! n=3");
    }

    #[test]
    fn template_substitution_with_step() {
        let inputs = Value::Null;
        let mut steps = BTreeMap::new();
        steps.insert(
            "s".to_owned(),
            StepResult {
                exit_code: 7,
                ..StepResult::default()
            },
        );
        let c = Value::Null;
        let e = BTreeMap::new();
        let ctx = make_ctx(&inputs, &steps, &c, &e);
        let out = eval_string("exit=${{ steps.s.exit_code }}", &ctx).unwrap();
        assert_eq!(out, "exit=7");
    }

    #[test]
    fn template_no_substitution() {
        let inputs = Value::Null;
        let s = BTreeMap::new();
        let c = Value::Null;
        let e = BTreeMap::new();
        let ctx = make_ctx(&inputs, &s, &c, &e);
        let out = eval_string("plain text", &ctx).unwrap();
        assert_eq!(out, "plain text");
    }

    #[test]
    fn unclosed_template_errors() {
        let inputs = Value::Null;
        let s = BTreeMap::new();
        let c = Value::Null;
        let e = BTreeMap::new();
        let ctx = make_ctx(&inputs, &s, &c, &e);
        let err = eval_string("${{ inputs.x", &ctx).unwrap_err();
        assert!(err.to_string().contains("unclosed"));
    }

    /// `find_close` must skip `\"` inside a `"`-delimited string
    /// literal, otherwise the embedded escape would be mistaken for a
    /// string close and the `}}` after the literal would never be
    /// found.
    #[test]
    fn template_with_escaped_quote_in_string_literal() {
        let inputs = Value::Null;
        let s = BTreeMap::new();
        let c = Value::Null;
        let e = BTreeMap::new();
        let ctx = make_ctx(&inputs, &s, &c, &e);
        // Source: `${{ "a\"b" }}` -> the parsed string literal is
        // `a"b`, rendered bare into the output.
        let out = eval_string(r#"${{ "a\"b" }}"#, &ctx).unwrap();
        assert_eq!(out, "a\"b");
    }

    #[test]
    fn truthiness_rules() {
        let (i, s, c, e) = ctx_with_inputs(json!({"empty_str": "", "zero": 0, "arr": []}));
        let ctx = make_ctx(&i, &s, &c, &e);
        assert!(!eval_bool("inputs.empty_str", &ctx).unwrap());
        assert!(!eval_bool("inputs.zero", &ctx).unwrap());
        assert!(!eval_bool("inputs.arr", &ctx).unwrap());
        assert!(eval_bool("\"x\"", &ctx).unwrap());
        assert!(eval_bool("1", &ctx).unwrap());
    }

    #[test]
    fn trailing_chars_error() {
        let (i, s, c, e) = ctx_with_inputs(Value::Null);
        let ctx = make_ctx(&i, &s, &c, &e);
        let err = eval_value("1 2", &ctx).unwrap_err();
        assert!(err.to_string().contains("trailing"));
    }

    #[test]
    fn parens_grouping() {
        let (i, s, c, e) = ctx_with_inputs(Value::Null);
        let ctx = make_ctx(&i, &s, &c, &e);
        assert!(eval_bool("(1 == 1) && (2 < 3)", &ctx).unwrap());
    }

    /// Errors include 1-based line/col alongside the byte offset, so
    /// users can locate the offending character in multi-line YAML
    /// without counting bytes.
    #[test]
    fn error_messages_include_line_col() {
        let (i, s, c, e) = ctx_with_inputs(Value::Null);
        let ctx = make_ctx(&i, &s, &c, &e);
        let err = eval_value("@", &ctx).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("line 1"), "{msg}");
        assert!(msg.contains("col 1"), "{msg}");
        assert!(msg.contains("byte 0"), "{msg}");
    }

    #[test]
    fn line_col_helper_handles_multiline() {
        // Three lines, byte indices: a=0, b=1, \n=2, c=3, d=4, \n=5,
        // e=6, f=7, g=8. So pos 7 -> 'f', which sits at line 3 col 2.
        let src = "ab\ncd\nefg";
        let (line, col) = line_col(src, 7);
        assert_eq!((line, col), (3, 2));
    }
}
