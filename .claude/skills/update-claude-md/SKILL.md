# Skill: update_claude_md

Update CLAUDE.md and docs/ reference files based on recent work in this session.

## When to use

Run `/update_claude_md` at the end of a work session to capture any architectural, structural, or important changes made during the session.

## Instructions

Follow these steps exactly:

### Step 1: Gather recent changes

1. Run `git diff --stat` and `git diff --name-only` to see what files were modified/added/removed in this session.
2. Run `git status` to see uncommitted changes.
3. Review the conversation history for what was done (new features, refactors, config changes, new files, deleted files, new commands, new constraints discovered).

### Step 2: Read current state

1. Read `CLAUDE.md` fully.
2. Skim `docs/architecture.md` (for file tree accuracy).
3. Identify which docs/ files might need updates based on what changed.

### Step 3: Classify changes

For each change found, classify it:

- **CORE** (goes in CLAUDE.md): File tree changes, new/removed commands, new code constraints, config changes, signal theory updates, eval checklist changes, canonical algorithm changes.
- **DETAIL** (goes in docs/): New experiment results → `docs/ml-pipeline.md`. Trading bot changes → `docs/trading-bot.md`. Telegram changes → `docs/telegram-bot.md`. Deployment changes → `docs/deployment.md`. Pipeline changes → `docs/pipeline-alignment.md`. Architecture detail → `docs/architecture.md`.
- **MEMORY** (goes in MEMORY.md): Operational discoveries, debugging insights, user preferences, active TODOs.
- **SKIP**: Routine bug fixes, minor refactors, formatting changes — no doc update needed.

### Step 4: Apply updates

For CORE changes:
- Update the relevant section of CLAUDE.md (keep it under ~100 lines total).
- If adding detail would bloat CLAUDE.md, put the detail in the appropriate docs/ file and add/update a path reference in CLAUDE.md's reference table.

For DETAIL changes:
- Update the appropriate docs/ file.
- Ensure CLAUDE.md reference table includes the path.

For MEMORY changes:
- Update MEMORY.md (keep it under ~50 lines).

**CRITICAL RULES:**
- CLAUDE.md must stay LEAN. If in doubt, put it in docs/ and reference it.
- Never duplicate information across CLAUDE.md and docs/ — one canonical location per fact.
- The file tree in `docs/architecture.md` must reflect actual current files (check with `ls` if unsure).
- Remove outdated information rather than accumulating it.

### Step 5: Log the update

Append an entry to `docs/update_log.md` in this format:

```
## YYYY-MM-DD

**Changes detected**: [list of what changed, or "None"]
**Files updated**: [list of files modified, or "No updates needed"]
**Summary**: [1-2 sentence summary]
```

If nothing needs updating, still log it:

```
## YYYY-MM-DD

**Changes detected**: None
**Files updated**: No updates needed
**Summary**: CLAUDE.md and docs/ are current. No changes from this session.
```

### Step 6: Report to user

Print a concise summary:
- What was updated (or "Everything is fresh — no updates needed")
- Current CLAUDE.md line count
- Any warnings (e.g., "CLAUDE.md is over 100 lines, consider trimming")
