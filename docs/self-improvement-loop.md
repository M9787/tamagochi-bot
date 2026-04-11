# Self-Improvement Loop

Reflect-Abstract-Rule loop that captures correction signals during a session and consolidates them into persistent rules for future sessions. Fully wired and empirically validated (11-test suite, all pass, zero drift).

## Components

### Capture side -- hooks (`.claude/hooks/`)

| Hook | Event | Purpose |
|---|---|---|
| `correction_detector.cjs` | UserPromptSubmit | Regex-scan prompt for 3 signal types. Append JSONL to `.claude/session_corrections.jsonl`. Never blocks. |
| `failure_logger.cjs` | PostToolUse (matcher: `Bash\|Edit\|Write\|Read\|Grep\|Glob`) | Extract errors from `tool_response` (is_error / stderr / non-zero exit). Append `signal_type: tool_failure`. |
| `stop_reflector.cjs` | Stop | Count queued signals for current session. If >=1, inject `additionalContext` reminder to run `/reflect` at next natural break. Also checks LEARNINGS.md / MEMORY.md bloat thresholds. |

Registered in `.claude/settings.json`.

### Signal patterns (correction_detector)

```js
{ type: "correction",     re: /\b(no|don'?t|stop|wrong|bad|never|why did you|you always|you should have)\b/i }
{ type: "confirmation",   re: /\b(yes exactly|perfect|keep doing that|good call|that was right)\b/i }
{ type: "meta_complaint", re: /\b(same mistake|again|like last time)\b/i }
```

**Known false positive**: `/\bagain\b/i` matches neutral phrases like "once again" / "try again". Candidate tightening: `/(same\s+mistake|like\s+last\s+time|did it again)/i`. Not yet applied.

### Bloat thresholds (stop_reflector)

| Constant | Value |
|---|---|
| `LEARNINGS_MAX_ENTRIES` | 50 |
| `LEARNINGS_MAX_LINES` | 300 |
| `MEMORY_MAX_ENTRIES` | 15 |
| `DUPLICATE_STEM_THRESHOLD` | 3 |

Alert fires via the same `additionalContext` mechanism when thresholds are exceeded.

### Consumer side -- skills (`.claude/skills/`)

| Skill | Role |
|---|---|
| `reflect/SKILL.md` | Read queue, group by theme, read existing LEARNINGS/MEMORY for dedup, draft Reflect-Abstract-Rule candidates, present via AskUserQuestion, write approved entries, clear queue. **Never writes without user approval.** |
| `learn/SKILL.md` | Manual entry point: "remember this lesson" style invocation. |
| `consolidate/SKILL.md` | Prune / merge / archive. Invoked when bloat thresholds fire. |

### Persistent store

- **`.claude/LEARNINGS.md`** -- Git-tracked, auto-loaded at session start via `@.claude/LEARNINGS.md` import in `CLAUDE.md`. Reflect-Abstract-Rule format only. Hard Constraints section capped at 10 absolutes.
- **`.claude/LEARNINGS_archive.md`** -- Pruned entries from `consolidate`.
- **`~/.claude/projects/.../memory/MEMORY.md`** -- User-specific auto-memory (working style, preferences). Separate from project-wide LEARNINGS.

## Flow

```
User prompt
  -> correction_detector.cjs (UserPromptSubmit)
       -> session_corrections.jsonl [signal captured]
Tool call
  -> failure_logger.cjs (PostToolUse)
       -> session_corrections.jsonl [failure captured]
End of turn (Stop event)
  -> stop_reflector.cjs
       -> count session signals
       -> inject additionalContext: "run `reflect` at next natural break"
       -> (optional) bloat alert
Next natural break
  -> /reflect skill
       -> read queue, dedup vs LEARNINGS/MEMORY
       -> AskUserQuestion for approval
       -> write approved entries
       -> clear queue
```

## Validation

11-test empirical suite executed 2026-04-11 with baseline SHA256 snapshots of both `session_corrections.jsonl` and `LEARNINGS.md`. All tests pass; both files restored byte-identical after cleanup.

| # | Test | Result |
|---|---|---|
| T0 | Baseline snapshot | - |
| T1 | correction_detector "stop doing that" -> `correction` | PASS |
| T2 | correction_detector "yes exactly perfect" -> `confirmation` | PASS |
| T3 | correction_detector "same mistake again" -> `meta_complaint` | PASS |
| T4 | correction_detector "what time is it" -> no match | PASS |
| T5 | failure_logger synthetic `is_error:true` -> `tool_failure` | PASS |
| T6 | failure_logger success -> no append | PASS |
| T7 | stop_reflector TEST_SID count=4 -> reminder injected | PASS |
| T8 | stop_reflector no bloat alert (1 entry, 6 memory) | PASS |
| T9 | Cleanup -> both SHA256 equal T0 | PASS |
| T10 | `/reflect` skill loads, classifies 3 existing entries as false-positive noise, no write | PASS |

## Do / Don't

- **Do** run `/reflect` manually when the Stop hook injects the reminder.
- **Do** review candidates before approval -- the skill will not write without explicit "Keep".
- **Don't** write to `LEARNINGS.md` directly; always go through the skill flow.
- **Don't** add rules a linter or type-checker could enforce.
- **Don't** duplicate rules; update in place.
