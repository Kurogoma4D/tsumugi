//! Integration test: register tools, dispatch by name, execute, verify results.

#![expect(clippy::unwrap_used, reason = "integration test assertions")]

use tmg_tools::{ToolRegistry, default_registry};

#[tokio::test]
async fn dispatch_file_write_then_read() {
    let registry = default_registry();

    let dir = std::env::temp_dir().join("tmg_tools_integration_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("hello.txt");

    // Write a file via registry dispatch.
    let write_result = registry
        .execute(
            "file_write",
            serde_json::json!({
                "path": file.to_str().unwrap(),
                "content": "integration test content"
            }),
        )
        .await
        .unwrap();

    assert!(!write_result.is_error);
    assert!(write_result.output.contains("Successfully wrote"));

    // Read it back via registry dispatch.
    let read_result = registry
        .execute(
            "file_read",
            serde_json::json!({ "path": file.to_str().unwrap() }),
        )
        .await
        .unwrap();

    assert!(!read_result.is_error);
    assert!(read_result.output.contains("integration test content"));

    // Patch the file via registry dispatch.
    let patch_result = registry
        .execute(
            "file_patch",
            serde_json::json!({
                "path": file.to_str().unwrap(),
                "old_string": "integration test",
                "new_string": "patched"
            }),
        )
        .await
        .unwrap();

    assert!(!patch_result.is_error);

    // Verify patch via re-read.
    let read_result2 = registry
        .execute(
            "file_read",
            serde_json::json!({ "path": file.to_str().unwrap() }),
        )
        .await
        .unwrap();

    assert!(read_result2.output.contains("patched content"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn dispatch_list_dir() {
    let registry = default_registry();

    let dir = std::env::temp_dir().join("tmg_tools_integration_list");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("subdir")).unwrap();
    std::fs::write(dir.join("file.txt"), "content").unwrap();

    let result = registry
        .execute(
            "list_dir",
            serde_json::json!({ "path": dir.to_str().unwrap() }),
        )
        .await
        .unwrap();

    assert!(!result.is_error);
    assert!(result.output.contains("file.txt"));
    assert!(result.output.contains("subdir/"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn dispatch_grep_search() {
    let registry = default_registry();

    let dir = std::env::temp_dir().join("tmg_tools_integration_grep");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("test.rs"), "fn main() {\n    println!(\"hello\");\n}\n").unwrap();

    let result = registry
        .execute(
            "grep_search",
            serde_json::json!({
                "pattern": "fn main",
                "path": dir.to_str().unwrap()
            }),
        )
        .await
        .unwrap();

    assert!(!result.is_error);
    assert!(result.output.contains("fn main"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn dispatch_shell_exec() {
    let registry = default_registry();

    let result = registry
        .execute(
            "shell_exec",
            serde_json::json!({ "command": "echo integration_test" }),
        )
        .await
        .unwrap();

    assert!(!result.is_error);
    assert!(result.output.contains("integration_test"));
}

#[tokio::test]
async fn dispatch_nonexistent_tool() {
    let registry = default_registry();

    let result = registry
        .execute("nonexistent_tool", serde_json::json!({}))
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not found"));
}

#[test]
fn all_schemas_returns_all_tools() {
    let registry = default_registry();
    let schemas = registry.all_schemas();

    let arr = schemas.as_array().unwrap();
    assert_eq!(arr.len(), 6);

    let names: Vec<&str> = arr
        .iter()
        .filter_map(|s| s.get("name").and_then(serde_json::Value::as_str))
        .collect();

    assert!(names.contains(&"file_read"));
    assert!(names.contains(&"file_write"));
    assert!(names.contains(&"file_patch"));
    assert!(names.contains(&"list_dir"));
    assert!(names.contains(&"grep_search"));
    assert!(names.contains(&"shell_exec"));

    // Each schema should have parameters.
    for schema in arr {
        assert!(schema.get("parameters").is_some());
        assert!(schema.get("description").is_some());
    }
}

#[test]
fn registry_get_by_name() {
    let registry = default_registry();

    assert!(registry.get("file_read").is_some());
    assert!(registry.get("nonexistent").is_none());
}

#[test]
fn custom_tool_registration() {
    use std::future::Future;
    use std::pin::Pin;
    use tmg_tools::{Tool, ToolError, ToolResult};

    struct CustomTool;

    impl Tool for CustomTool {
        fn name(&self) -> &'static str {
            "custom"
        }

        fn description(&self) -> &'static str {
            "A custom test tool."
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({ "type": "object" })
        }

        fn execute(
            &self,
            _params: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolResult, ToolError>> + Send + '_>> {
            Box::pin(async { Ok(ToolResult::success("custom output")) })
        }
    }

    let mut registry = ToolRegistry::new();
    registry.register(CustomTool);

    assert!(registry.get("custom").is_some());
    assert_eq!(registry.get("custom").unwrap().name(), "custom");
}
