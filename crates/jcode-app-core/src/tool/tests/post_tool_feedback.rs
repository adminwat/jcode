use super::*;

#[cfg(unix)]
#[tokio::test]
async fn post_tool_feedback_hook_appends_stdout_to_tool_result() {
    use std::os::unix::fs::PermissionsExt;
    let _env_lock = crate::storage::lock_test_env();
    let dir = tempfile::tempdir().expect("temp dir");
    let target = dir.path().join("prose.md");
    std::fs::write(&target, "hello\n").expect("write target");
    let script = dir.path().join("feedback.sh");
    std::fs::write(
        &script,
        "#!/bin/sh\ninput=$(cat)\necho \"call=$JCODE_HOOK_TOOL_CALL_ID tool=$JCODE_HOOK_TOOL_NAME\"\necho \"paths=$JCODE_HOOK_TOUCHED_PATHS\"\ncase \"$input\" in *prose.md*) echo stdin-ok ;; esac\nexit 1\n",
    )
    .expect("write script");
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    let previous = std::env::var_os("JCODE_HOOK_POST_TOOL_FEEDBACK");
    crate::env::set_var("JCODE_HOOK_POST_TOOL_FEEDBACK", &script);
    crate::config::invalidate_config_cache(); // throttled cache: apply the env now

    let registry = Registry::new(Arc::new(MockProvider) as Arc<dyn Provider>).await;
    let mut ctx = mcp_test_context(dir.path());
    ctx.tool_call_id = "call_feedback".to_string();
    let result = registry
        .execute("read", serde_json::json!({"file_path": "prose.md"}), ctx)
        .await;

    match previous {
        Some(value) => crate::env::set_var("JCODE_HOOK_POST_TOOL_FEEDBACK", value),
        None => crate::env::remove_var("JCODE_HOOK_POST_TOOL_FEEDBACK"),
    }
    let output = result.expect("read should succeed").output;
    assert!(output.contains("hello"), "{output}");
    assert!(output.contains("[post_tool_feedback]"), "{output}");
    assert!(output.contains("call=call_feedback tool=read"), "{output}");
    assert!(
        output.contains(&format!("paths={}", target.display())),
        "{output}"
    );
    assert!(output.contains("stdin-ok"), "{output}");
}
