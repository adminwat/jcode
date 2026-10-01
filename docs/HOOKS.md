# Lifecycle Hooks

jcode can run external commands at well-defined lifecycle points so other
programs can observe or gate agent behavior without forking jcode. Hooks
complement the [spawn hook](SPAWN_HOOK.md) (which controls *where headed
sessions appear*); lifecycle hooks tell you *what is happening inside them*.

## Configuration

```toml
# ~/.jcode/config.toml
[hooks]
turn_start    = ""                            # observer
turn_context  = ""                            # bounded context response
turn_end      = "~/bin/jcode-turn-notify"     # observer
session_start = ""                            # observer
session_end   = ""                            # observer
pre_tool      = "~/bin/jcode-tool-policy"     # gate
post_tool     = ""                            # observer
pre_tool_timeout_ms = 5000
```

Env overrides (always win; empty value disables a config hook):
`JCODE_HOOK_TURN_START`, `JCODE_HOOK_TURN_CONTEXT`, `JCODE_HOOK_TURN_END`, `JCODE_HOOK_SESSION_START`, `JCODE_HOOK_SESSION_END`,
`JCODE_HOOK_PRE_TOOL`, `JCODE_HOOK_POST_TOOL`, `JCODE_HOOK_PRE_TOOL_TIMEOUT_MS`.

## Common contract

- The hook command line is parsed shell-style (quotes and backslash escapes
  work) but executed **directly**, not through a shell. A leading `~/` in the
  program path is expanded.
- The hook runs in the session working directory when known.
- Every hook receives:

| Variable | Meaning |
| --- | --- |
| `JCODE_HOOK_EVENT` | `turn_start`, `turn_context`, `turn_end`, `session_start`, `session_end`, `pre_tool`, `post_tool` |
| `JCODE_HOOK_SESSION_ID` | Session the event belongs to |
| `JCODE_HOOK_CWD` | Session working directory |
| `JCODE_HOOK_PAYLOAD` | JSON object mirroring all fields (capped at 16 KB) |
| `JCODE_HOOKS_DISABLED` | Always `1`; suppresses hooks in nested jcode calls (recursion guard) |

## Observer hooks

`turn_start`, `turn_end`, `session_start`, `session_end`, and `post_tool` are
**observers**: spawned detached, fire-and-forget. They can never block or slow
the agent; failures are only logged.

### `turn_end`

Fires when an agent turn completes (streaming turn path, which covers TUI,
desktop, swarm workers, and headless sessions).

Extra fields: `JCODE_HOOK_STATUS` (`ok`/`error`), `JCODE_HOOK_DURATION_MS`,
`JCODE_HOOK_MODEL`, `JCODE_HOOK_LAST_ASSISTANT_TEXT` (first 4000 chars),
`JCODE_HOOK_ERROR` (on failure).

### `session_start` / `session_end`

`session_start` fires when an agent session becomes active, with
`JCODE_HOOK_SOURCE` = `create` (brand new), `attach` (existing session object
attached), or `resume` (restored by id). `session_end` fires on normal close
(`JCODE_HOOK_SOURCE=close`).

### `post_tool`

Fires after every tool call. Extra fields: `JCODE_HOOK_TOOL_NAME`,
`JCODE_HOOK_STATUS`, `JCODE_HOOK_DURATION_MS`, `JCODE_HOOK_OUTPUT_BYTES` (on
success), `JCODE_HOOK_ERROR` (on failure).

## Context response hook: `turn_context`

This protocol is opt-in and distinct from detached observers. Agent blocking and
streaming provider loops await it before the first request, then replay the same
evidence during tool continuations. The query comes from the current turn's
durable user-message anchor, including on resume, never a later tool result.
Abstentions and failures are cached for that turn too. A new user turn queries
again. Disabling memory or hooks suppresses delivery, and session reset/close/crash
clears the Agent cache. External context is always ephemeral, even when native
memory persistence is enabled. Source and record IDs are logged, not record text.
Streaming clients receive one memory event per prepared nonempty result.
This does not configure external-store readers or change existing installations.

A scalar command or array of up to four commands is supported. Commands run in
parallel within one **1500ms deadline**, including stdin writes, stdout reads,
and exit. Timeout, cancellation, or protocol errors discard that command's
context without failing the user turn. On Unix the dedicated hook process group
is killed on completion or cancellation, including descendants. Direct children
also use Tokio's kill-on-drop/reaping. No store or model is invoked by default.

The complete UTF-8 JSON request is sent on **stdin**, not in an environment variable:

```json
{"version":1,"event":"turn_context","session_id":"full-session-id","turn_id":"logical-turn-id","cwd":"/project","query":"actual user query"}
```

`cwd` may be null. `JCODE_HOOK_TURN_ID` is also provided, alongside the common
metadata and recursion guard. `JCODE_HOOK_PAYLOAD` contains metadata only, not
the query. Requests over 64 KiB, blank identity/query, and configurations over
four commands are rejected, never silently truncated.

Exit zero and return exactly one JSON object on **stdout**:

```json
{"version":1,"session_id":"full-session-id","turn_id":"logical-turn-id","memories":[{"source":"claude-mem","id":"observation-123","text":"Historical evidence with attribution."}]}
```

Both identities must match the request exactly. Unknown/missing fields, invalid
JSON, nonzero exit, or stdout exceeding 16 KiB reject the entire response. An
empty `memories` array is a valid abstention, but `OK`, `{}`, or an HTTP success
alone are not. Each record requires nonblank `source` (at most 128 bytes), `id`
(1024 bytes), and `text` (8192 bytes), with at most 32 records per response.
The combined serialized record arrays are bounded to 16 KiB in configuration
order. Contributions exceeding the remaining budget are rejected and logged,
not cut mid-evidence. Identical records are deduplicated. Diagnostic logs contain
status, record count and latency, not queries or returned evidence.

Store output is historical, untrusted evidence, not instructions or proof of
current state. Consumers must preserve provenance, bind replay to this logical
turn, and keep retrieval out of persisted conversation history.

## Gate hook: `pre_tool`

`pre_tool` runs **synchronously before every tool call** and can block it:

- The hook receives `JCODE_HOOK_TOOL_NAME` plus the full tool input JSON on
  **stdin** (and a 16 KB-truncated copy in `JCODE_HOOK_TOOL_INPUT`).
- **Exit 0**: allow the call.
- **Exit 2**: block the call. The hook's stderr (trimmed, capped at 2000
  chars) is returned to the model as the tool error, so the model can adapt.
- **Anything else fails open** with a logged warning: other exit codes,
  timeout (`pre_tool_timeout_ms`, default 5s), missing binary, spawn errors.

Fail-open is deliberate: a broken policy script should degrade to "no policy"
rather than brick every session. If you need fail-closed semantics, make the
hook itself robust (it is your trust boundary, not jcode).

### Example policy script

```bash
#!/usr/bin/env bash
# ~/bin/jcode-tool-policy
# stdin: tool input JSON. Env: JCODE_HOOK_TOOL_NAME, JCODE_HOOK_SESSION_ID...
input=$(cat)

case "$JCODE_HOOK_TOOL_NAME" in
  bash)
    if grep -qE 'rm -rf /([^a-zA-Z]|$)|mkfs|dd if=' <<<"$input"; then
      echo "blocked: destructive shell command" >&2
      exit 2
    fi
    ;;
  write|edit)
    if grep -q '"file_path":"/etc/' <<<"$input"; then
      echo "blocked: writes to /etc are not allowed" >&2
      exit 2
    fi
    ;;
esac
exit 0
```

## Example: tmux status + desktop notification on turn end

```bash
#!/usr/bin/env bash
# ~/bin/jcode-turn-notify
if [ "$JCODE_HOOK_STATUS" = ok ]; then icon=✅; else icon=❌; fi
tmux display-message "jcode $icon ${JCODE_HOOK_SESSION_ID:0:12}" 2>/dev/null
notify-send "jcode turn $JCODE_HOOK_STATUS" \
  "${JCODE_HOOK_LAST_ASSISTANT_TEXT:0:120}" 2>/dev/null
exit 0
```

## Example: JSON event log of all hook activity

Point several hooks at one script and fan out on `JCODE_HOOK_EVENT`:

```bash
#!/usr/bin/env bash
# ~/bin/jcode-event-log
echo "$JCODE_HOOK_PAYLOAD" >> ~/.local/state/jcode-events.jsonl
```

```toml
[hooks]
turn_end      = "~/bin/jcode-event-log"
session_start = "~/bin/jcode-event-log"
session_end   = "~/bin/jcode-event-log"
post_tool     = "~/bin/jcode-event-log"
```

## Design notes

- Hook lookups are config-driven and re-read on config reload; you can add or
  change hooks without restarting jcode.
- Hot paths (`pre_tool`/`post_tool`) check whether a hook is configured before
  building any payload, so unconfigured hooks cost ~nothing.
- The recursion guard (`JCODE_HOOKS_DISABLED=1`) means a hook may safely call
  `jcode` CLI commands without re-triggering hooks in that nested process.
