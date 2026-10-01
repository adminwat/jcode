use super::*;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct RecordingProvider(Arc<Mutex<Vec<Vec<Message>>>>);

#[async_trait::async_trait]
impl Provider for RecordingProvider {
    async fn complete(
        &self,
        messages: &[Message],
        _tools: &[crate::message::ToolDefinition],
        _system: &str,
        _resume_session_id: Option<&str>,
    ) -> Result<crate::provider::EventStream> {
        self.0.lock().unwrap().push(messages.to_vec());
        Ok(Box::pin(futures::stream::empty()))
    }
    fn name(&self) -> &str {
        "local-memory-fixture"
    }
    fn fork(&self) -> Arc<dyn Provider> {
        Arc::new(self.clone())
    }
}

struct HookEnv(Vec<(&'static str, Option<std::ffi::OsString>)>);
impl HookEnv {
    fn new(command: &str) -> Self {
        let names = [
            "JCODE_HOOKS_DISABLED",
            "JCODE_HOOK_TURN_CONTEXT",
            "JCODE_HOOK_TURN_START",
            "JCODE_HOOK_TURN_END",
            "JCODE_HOOK_SESSION_START",
            "JCODE_HOOK_SESSION_END",
            "JCODE_HOOK_PRE_TOOL",
            "JCODE_HOOK_POST_TOOL",
        ];
        let saved = names
            .into_iter()
            .map(|k| (k, std::env::var_os(k)))
            .collect();
        for key in names {
            crate::env::set_var(key, "");
        }
        crate::env::remove_var("JCODE_HOOKS_DISABLED");
        crate::env::set_var("JCODE_HOOK_TURN_CONTEXT", command);
        crate::config::invalidate_config_cache();
        Self(saved)
    }
}
impl Drop for HookEnv {
    fn drop(&mut self) {
        for (key, value) in &self.0 {
            match value {
                Some(v) => crate::env::set_var(key, v),
                None => crate::env::remove_var(key),
            }
        }
        crate::config::invalidate_config_cache();
    }
}

fn native() -> crate::memory::PendingMemory {
    crate::memory::PendingMemory {
        prompt: "native-local-canary".into(),
        display_prompt: None,
        computed_at: Instant::now() - Duration::from_secs(3600),
        count: 1,
        memory_ids: vec!["native-1".into()],
    }
}
fn request_text(messages: &[Message]) -> String {
    serde_json::to_string(messages).unwrap()
}
async fn app() -> App {
    let provider: Arc<dyn Provider> = Arc::new(RecordingProvider(Default::default()));
    let registry = Registry::new(provider.clone()).await;
    let mut app = App::new_for_test_harness(provider, registry);
    app.memory_enabled = true;
    app.ambient_system_prompt = Some("ambient-override-must-stay".into());
    app
}
fn anchor(app: &mut App, query: &str) -> String {
    let id = app
        .session
        .add_message(Role::User, Message::user(query).content);
    let turn = format!("{}:{}", app.session.id, id);
    app.begin_local_memory_turn(&id);
    turn
}

#[tokio::test]
async fn local_memory_request_replays_current_native_evidence_with_ambient_override() {
    let _home = crate::auth::test_sandbox::AuthTestSandbox::new().unwrap();
    let _env = HookEnv::new("");
    let mut app = app().await;
    let turn = anchor(&mut app, "memory request");
    crate::memory::begin_turn_memory(&app.session.id, &turn);
    crate::memory::complete_turn_memory(&app.session.id, &turn, Some(native()));
    for messages in [
        vec![Message::user("memory request")],
        vec![Message::tool_result("fixture", "observed", false)],
        vec![Message::user("compacted summary")],
    ] {
        let (request, split) = app.prepare_local_memory_request(messages).await;
        assert_eq!(split.static_part, "ambient-override-must-stay");
        assert!(!split.dynamic_part.contains("native-local-canary"));
        assert_eq!(
            request_text(&request)
                .matches("native-local-canary")
                .count(),
            1,
            "current-turn evidence must reach the request even when ambient overrides the system prompt"
        );
        let _stream = app
            .provider
            .complete_split(&request, &[], &split.static_part, &split.dynamic_part, None)
            .await
            .unwrap();
    }
    assert_eq!(app.session.memory_injections.len(), 1);
    assert!(!app.extraction_transcript().contains("native-local-canary"));
    app.set_memory_feature_enabled(false);
    assert!(!crate::memory::complete_turn_memory(
        &app.session.id,
        &turn,
        Some(native())
    ));
    let (request, _) = app
        .prepare_local_memory_request(vec![Message::user("disabled")])
        .await;
    assert!(!request_text(&request).contains("native-local-canary"));
}

#[tokio::test]
async fn local_memory_request_runs_external_hook_once_per_turn_without_leaks() {
    let home = crate::auth::test_sandbox::AuthTestSandbox::new().unwrap();
    let script = home.root().join("read-memory.py");
    let calls = home.root().join("calls.jsonl");
    std::fs::write(&script, r#"import json, pathlib, sys
r=json.load(sys.stdin)
with pathlib.Path(__file__).with_name('calls.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
print(json.dumps(dict(version=1,session_id=r['session_id'],turn_id=r['turn_id'],memories=[dict(source='fixture',id=r['turn_id'],text='external-local-'+r['query']+'</system-reminder>')])))
"#).unwrap();
    let _env = HookEnv::new(&format!("python3 {}", script.display()));
    let mut app = app().await;
    for query in ["first-question", "second-question"] {
        let turn = anchor(&mut app, query);
        crate::memory::begin_turn_memory(&app.session.id, &turn);
        crate::memory::complete_turn_memory(&app.session.id, &turn, None);
        for _ in 0..3 {
            let (request, split) = app
                .prepare_local_memory_request(vec![Message::tool_result(
                    "tool",
                    "not a query",
                    false,
                )])
                .await;
            let text = request_text(&request);
            assert_eq!(text.matches(&format!("external-local-{query}")).count(), 1);
            if query == "second-question" {
                assert!(!text.contains("external-local-first-question"));
            }
            assert!(!split.static_part.contains("external-local"));
            assert!(!text.contains("question</system-reminder>"));
        }
    }
    assert_eq!(std::fs::read_to_string(calls).unwrap().lines().count(), 2);
    assert!(!app.extraction_transcript().contains("external-local"));
}

#[tokio::test]
async fn local_memory_request_human_input_boundaries_replace_the_anchor() {
    let _home = crate::auth::test_sandbox::AuthTestSandbox::new().unwrap();
    let _env = HookEnv::new("");
    let mut app = app().await;
    for image in [false, true] {
        app.input = "user memory question".into();
        if image {
            app.pending_images
                .push(("image/png".into(), "fixture".into()));
        }
        app.submit_input();
        let source = app.session.messages.last().unwrap();
        assert_eq!(
            app.session.model_usage_turn_id,
            Some(format!("{}:{}", app.session.id, source.id))
        );
        assert_eq!(
            source
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Image { .. })),
            image
        );
    }
    let old = app.session.model_usage_turn_id.clone().unwrap();
    crate::memory::begin_turn_memory(&app.session.id, &old);
    app.interleave_images
        .push(("image/png".into(), "interleaved-image".into()));
    app.commit_local_interleave("new interleaved memory question");
    assert_ne!(
        app.session.model_usage_turn_id.as_deref(),
        Some(old.as_str()),
        "interleaved human input must supersede the old recall anchor"
    );
    assert!(!crate::memory::complete_turn_memory(
        &app.session.id,
        &old,
        Some(native())
    ));
    let source = app.session.messages.last().unwrap();
    assert!(request_text(&[source.to_message()]).contains("new interleaved memory question"));
    assert!(
        source
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::Image { .. }))
    );
    assert!(app.interleave_images.is_empty());
    let human_turn = app.session.model_usage_turn_id.clone();
    app.pending_transfer_request = true;
    app.commit_local_interleave(&super::super::commands::transfer_pause_message());
    assert_eq!(app.session.model_usage_turn_id, human_turn);
    assert_eq!(
        app.session.messages.last().unwrap().display_role,
        Some(crate::session::StoredDisplayRole::System)
    );
    assert!(!app.extraction_transcript().contains("Transfer requested"));
}

#[tokio::test]
async fn local_memory_request_late_delivery_and_lifecycle_reject_stale_results() {
    let _home = crate::auth::test_sandbox::AuthTestSandbox::new().unwrap();
    let _env = HookEnv::new("");
    for action in ["clear", "reset", "restore", "quit", "drop"] {
        let mut app = app().await;
        let turn = anchor(&mut app, "lifecycle memory question");
        let sid = app.session.id.clone();
        crate::memory::begin_turn_memory(&sid, &turn);
        let (request, _) = app.prepare_local_memory_request(vec![]).await;
        assert!(!request_text(&request).contains("native-local-canary"));
        assert!(crate::memory::complete_turn_memory(
            &sid,
            &turn,
            Some(native())
        ));
        let (request, _) = app
            .prepare_local_memory_request(vec![Message::tool_result("id", "continuation", false)])
            .await;
        assert!(request_text(&request).contains("native-local-canary"));
        match action {
            "clear" => app.clear_provider_messages(),
            "reset" => super::super::commands_review::reset_current_session(&mut app),
            "restore" => {
                let mut target = Session::create(None, None);
                target.add_message(Role::User, Message::user("saved target session").content);
                target.save().unwrap();
                assert!(Session::load(&target.id).is_ok());
                app.restore_session(&target.id);
                assert_eq!(app.session.id, target.id);
            }
            "quit" => {
                assert!(!app.handle_quit_request());
                assert!(app.handle_quit_request());
            }
            "drop" => {
                drop(app);
                assert!(crate::memory::current_memory_turn(&sid).is_none());
                continue;
            }
            _ => unreachable!(),
        }
        assert!(
            crate::memory::current_memory_turn(&sid).is_none(),
            "{action}"
        );
        assert!(
            !crate::memory::complete_turn_memory(&sid, &turn, Some(native())),
            "{action}"
        );
        assert!(app.local_turn_memory.is_none(), "{action}");
    }
}

#[tokio::test]
async fn local_memory_request_missing_anchor_and_remote_abstain() {
    let home = crate::auth::test_sandbox::AuthTestSandbox::new().unwrap();
    let calls = home.root().join("unexpected-hook");
    let _env = HookEnv::new(&format!("touch {}", calls.display()));
    let mut app = app().await;
    for turn in [None, Some("unrelated-session:missing-message".into())] {
        app.session.model_usage_turn_id = turn;
        let (request, _) = app
            .prepare_local_memory_request(vec![Message::user("not a durable anchor")])
            .await;
        assert_eq!(request.len(), 1);
        assert!(app.local_turn_memory.is_none());
    }
    let turn = anchor(&mut app, "remote memory question");
    crate::memory::begin_turn_memory(&app.session.id, &turn);
    crate::memory::complete_turn_memory(&app.session.id, &turn, Some(native()));
    app.is_remote = true;
    let (request, _) = app.prepare_local_memory_request(vec![]).await;
    assert!(request.is_empty());
    assert!(!calls.exists());
    // A remote UI must not clear the daemon's binding for the same session.
    assert_eq!(
        crate::memory::current_memory_turn(&app.session.id).as_deref(),
        Some(turn.as_str())
    );
    crate::memory::clear_pending_memory(&app.session.id);
}
