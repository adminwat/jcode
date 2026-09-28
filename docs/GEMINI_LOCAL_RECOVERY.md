# Gemini local recovery verification

Verified 2026-09-28 against base `5007b8730`.

## Changes and observed behavior

| Requirement | Verification | Result |
| --- | --- | --- |
| Preserve OAuth quota/reset errors instead of reporting a later fallback 404 | Provider HTTP regression and rebuilt CLI with an isolated localhost Code Assist fixture, quota first and after an initial 404 | Both preserve quota, model and reset information. The installed baseline reproduced the misleading final 404. |
| Keep genuine model-not-found failures and successful fallback working | Rebuilt CLI fixture, all-404 and quota-then-success cases | Genuine 404 remains a failure. Successful fallback returns text and reports the actual fallback model. |
| Avoid automatic retries on long-horizon quota exhaustion | TUI classification and remote-event tests, with auto-poke enabled/disabled and reset metadata present/absent | No queued retry or reset timer. Auto-poke stops. Short burst rate limits remain retryable. |
| Avoid a stale generic 200k compaction budget for Gemini aliases or internal model fallback | Alias/cache tests and agent plus local-TUI model-switch tests | Gemini aliases use the existing family 1M window, cached overrides win, and compaction budgets track the provider window in both directions. No global cap/config change. |
| Explain missing files relative to the real tool directory | Read/edit/multiedit tests and rebuilt CLI read-tool round trip | Errors include the resolved path and explain that shell `cd` is call-local. Read suggestions remain intact. |
| Stay on OAuth | Every fixture request asserts bearer-token auth and rejects API-key auth | OAuth only. No real Google request or real credentials used by replay. |

Each new behavioral regression was checked against the original behavior before the fixed tests passed. The local-TUI fixture initially had a compilation error, which was corrected before observing the real 200k-versus-1M assertion failure and then the passing fixed test.

## Broader validation

- Provider core: 128 tests passed.
- Gemini runtime: 40 tests passed.
- Agent compaction: 5 tests passed.
- Gemini TUI focused tests: 5 passed.
- TUI auto-poke tests: 28 passed. Existing remote-event tests also passed.
- Tool suite: 46 passed, 2 failed, 1 ignored. The two failures are existing description-token-cap tests. Baseline and fixed logs contain the same 15 violations. An initially overlong new shell description was shortened rather than weakening the gate.
- Full-feature `selfdev` CLI build completed, including default PDF, embeddings and Bedrock features.
- Actual rebuilt CLI completed five isolated OAuth replay cases successfully. These are real-executable tests against synthetic server responses, not proof of recovery against Google's live quota.
- `git diff --check` passed. Mobile layout and raw-HTML checks do not apply to these CLI-only changes.

Local evidence: `~/.jcode/scratch/gemini-baseline-compare.SYf05o`, baseline executable replay `gemini-executable-replay-5ybcn_ho`, fixed executable replay `gemini-executable-replay-1lfx0wkp`.

## Deployment boundary

No authentication/config changes, transcript edits, global context cap or daemon restart are part of this fix. Active shared-server sessions remain on their existing binary until a separately verified safe activation. Building or publishing this candidate alone does not change those sessions. These fixes do not establish why Google's allowance was exhausted and do not prove that all cognition/looping symptoms are resolved.
