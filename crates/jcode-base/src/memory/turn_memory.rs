//! Request-bound handoff for asynchronous memory retrieval.
//! A retrieval belongs to one logical user turn, not the next turn to arrive.
use super::PendingMemory;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

#[derive(Default)]
struct TurnMemory {
    turn_id: String,
    ready: bool,
    delivered: bool,
    memory: Option<PendingMemory>,
}

static TURNS: LazyLock<Mutex<HashMap<String, TurnMemory>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Default)]
pub struct TurnMemoryResult {
    pub ready: bool,
    pub first_delivery: bool,
    pub memory: Option<PendingMemory>,
}

/// Returns true once per logical turn, when retrieval should be enqueued.
pub fn begin_turn_memory(session_id: &str, turn_id: &str) -> bool {
    let Ok(mut turns) = TURNS.lock() else {
        return false;
    };
    if turns
        .get(session_id)
        .is_some_and(|turn| turn.turn_id == turn_id)
    {
        return false;
    }
    turns.insert(
        session_id.to_owned(),
        TurnMemory {
            turn_id: turn_id.to_owned(),
            ..TurnMemory::default()
        },
    );
    true
}

/// Capture this identity before enqueuing asynchronous work, never on completion.
pub fn current_memory_turn(session_id: &str) -> Option<String> {
    TURNS
        .lock()
        .ok()?
        .get(session_id)
        .map(|turn| turn.turn_id.clone())
}

/// Returns false for a late result from a superseded turn or closed session.
pub fn complete_turn_memory(
    session_id: &str,
    turn_id: &str,
    memory: Option<PendingMemory>,
) -> bool {
    let Ok(mut turns) = TURNS.lock() else {
        return false;
    };
    let Some(turn) = turns.get_mut(session_id) else {
        return false;
    };
    if turn.turn_id != turn_id || turn.ready {
        return false;
    }
    turn.memory = memory;
    turn.ready = true;
    true
}

/// Replay ephemeral context on every continuation. Only the first read acknowledges delivery.
pub fn read_turn_memory(session_id: &str, turn_id: &str) -> TurnMemoryResult {
    let Ok(mut turns) = TURNS.lock() else {
        return TurnMemoryResult::default();
    };
    let Some(turn) = turns
        .get_mut(session_id)
        .filter(|turn| turn.turn_id == turn_id)
    else {
        return TurnMemoryResult::default();
    };
    let first_delivery = turn.memory.is_some() && !turn.delivered;
    turn.delivered |= first_delivery;
    TurnMemoryResult {
        ready: turn.ready,
        first_delivery,
        memory: turn.memory.clone(),
    }
}

pub fn clear_turn_memory(session_id: &str) {
    if let Ok(mut turns) = TURNS.lock() {
        turns.remove(session_id);
    }
}

pub(super) fn clear_all_turn_memory() {
    if let Ok(mut turns) = TURNS.lock() {
        turns.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn result(text: &str) -> PendingMemory {
        PendingMemory {
            prompt: text.into(),
            display_prompt: None,
            computed_at: Instant::now() - Duration::from_secs(3600),
            count: 1,
            memory_ids: vec!["fact-1".into()],
        }
    }

    #[test]
    fn turn_memory_rejects_old_turn_and_other_session() {
        let session = "turn_memory_identity";
        assert!(begin_turn_memory(session, "a"));
        assert!(!begin_turn_memory(session, "a"));
        assert_eq!(current_memory_turn(session).as_deref(), Some("a"));
        assert!(begin_turn_memory(session, "b"));
        assert!(!complete_turn_memory(session, "a", Some(result("old"))));
        assert!(!complete_turn_memory(
            "another_session",
            "b",
            Some(result("wrong"))
        ));
        assert!(read_turn_memory(session, "a").memory.is_none());
        assert!(!read_turn_memory(session, "b").ready);
        assert!(complete_turn_memory(session, "b", Some(result("right"))));
        assert_eq!(
            read_turn_memory(session, "b").memory.unwrap().prompt,
            "right"
        );
        clear_turn_memory(session);
    }

    #[test]
    fn turn_memory_replays_after_pause_but_acknowledges_once() {
        let session = "turn_memory_replay";
        begin_turn_memory(session, "a");
        assert!(complete_turn_memory(
            session,
            "a",
            Some(result("durable within turn"))
        ));
        let first = read_turn_memory(session, "a");
        assert!(first.ready && first.first_delivery);
        assert!(!first.memory.as_ref().unwrap().is_fresh());
        let continuation = read_turn_memory(session, "a");
        assert!(continuation.ready && !continuation.first_delivery);
        assert_eq!(
            first.memory.unwrap().prompt,
            continuation.memory.unwrap().prompt
        );
        assert!(!complete_turn_memory(
            session,
            "a",
            Some(result("duplicate producer"))
        ));
        assert!(begin_turn_memory(session, "b"));
        assert!(read_turn_memory(session, "b").memory.is_none());
        assert!(complete_turn_memory(session, "b", None));
        assert!(read_turn_memory(session, "b").ready);
        clear_turn_memory(session);
        assert!(current_memory_turn(session).is_none());
        assert!(!complete_turn_memory(session, "b", Some(result("closed"))));
    }
}
