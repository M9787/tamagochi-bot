# Subagent Factory Learnings
**Generated:** 2026-01-28
**Source:** Claude Code Documentation Library

---

## User Preferences

| Preference | Selection | Rationale |
|------------|-----------|-----------|
| Agent Types | Task-specific | Single-purpose, focused agents |
| Model Strategy | Sonnet default | Balance cost/quality, Opus for complex only |
| Tool Access | Minimal | Security-first, only essential tools |

---

## Subagent Structure Pattern

```markdown
---
name: {lowercase-hyphenated}
description: {Action verb} {what it does}. Use PROACTIVELY when {trigger condition}.
tools: {minimal-required-tools}
model: sonnet
---

{Focused system prompt - max 10 lines}
```

---

## Tool Selection Matrix

| Agent Purpose | Required Tools | Optional Tools |
|---------------|----------------|----------------|
| Code Review | Read, Grep, Glob | Bash(git diff *) |
| Testing | Bash, Read | Edit (for fixes) |
| Documentation | Read, Write, Glob | WebFetch |
| Debugging | Read, Grep, Bash | Edit |
| Security Audit | Read, Grep, Glob | - |
| Refactoring | Read, Edit, Glob | Bash(npm test) |
| Research/Explore | Read, Grep, Glob, WebSearch | WebFetch |

---

## Model Selection Rules

| Complexity | Model | Use When |
|------------|-------|----------|
| Low | haiku | Simple queries, validation, formatting |
| Medium | sonnet | Most tasks, code review, implementation |
| High | opus | Architecture, complex debugging, security |

**Default:** `sonnet` (unless task requires deep reasoning)

---

## Description Patterns

### Action-Oriented Triggers

| Pattern | Example |
|---------|---------|
| Proactive | `Use PROACTIVELY after code changes` |
| Mandatory | `MUST BE USED for all security-related code` |
| Conditional | `Use when debugging complex issues` |

### Verb Templates

- **Review:** `Reviews {target} for {concerns}`
- **Validate:** `Validates {target} against {criteria}`
- **Generate:** `Generates {output} from {input}`
- **Fix:** `Identifies and fixes {problem type}`
- **Explore:** `Investigates {domain} to understand {goal}`

---

## System Prompt Best Practices

### Structure (Max 10 lines)
```
Line 1: Role declaration
Lines 2-4: Core responsibilities (bullet list)
Lines 5-7: Output format requirements
Lines 8-10: Constraints/boundaries
```

### Anti-Patterns to Avoid
- Verbose explanations
- Redundant instructions
- Generic advice Claude already knows
- Multi-paragraph descriptions

---

## Minimal Token Templates

### Code Reviewer
```markdown
---
name: code-reviewer
description: Reviews code for quality issues. Use PROACTIVELY after Edit/Write.
tools: Read, Grep, Glob
model: sonnet
---
Senior engineer reviewing for:
- Logic errors and edge cases
- Performance concerns
- Code style violations
Output: Issue list with file:line references.
```

### Test Runner
```markdown
---
name: test-runner
description: Runs tests and reports failures. Use PROACTIVELY after code changes.
tools: Bash, Read
model: sonnet
---
Execute test suite, analyze failures.
- Run: `npm test` or detected test command
- Report: Failed tests with stack traces
- Suggest: Likely fix locations
```

### Security Scanner
```markdown
---
name: security-scanner
description: Scans for security vulnerabilities. MUST BE USED for auth/crypto code.
tools: Read, Grep, Glob
model: opus
---
Security audit for:
- Injection vulnerabilities
- Auth/authz flaws
- Secrets in code
- Insecure patterns
Output: CRITICAL/HIGH/MEDIUM findings with remediation.
```

### Documentation Writer
```markdown
---
name: doc-writer
description: Generates documentation from code. Use when docs requested.
tools: Read, Write, Glob
model: sonnet
---
Generate concise documentation:
- API: Function signatures + examples
- README: Setup, usage, architecture
- Comments: JSDoc/docstrings where missing
Match existing doc style in project.
```

---

## File Naming Convention

| Pattern | Example |
|---------|---------|
| Purpose-based | `code-reviewer.md`, `test-runner.md` |
| Domain-based | `react-specialist.md`, `python-debugger.md` |
| Action-based | `deploy-validator.md`, `pr-creator.md` |

**Rule:** Always lowercase, hyphen-separated, `.md` extension

---

## Validation Checklist

Before outputting any agent:

- [ ] Name is lowercase-hyphenated
- [ ] Description starts with action verb
- [ ] Description includes "Use PROACTIVELY" or "MUST BE USED"
- [ ] Tools list is minimal (only what's needed)
- [ ] Model is `sonnet` unless complex task requires `opus`
- [ ] System prompt is ≤10 lines
- [ ] No redundant information

---

## Q&A Log

**Q: What types of subagents do you primarily want to create?**
A: Task-specific (single-purpose agents)

**Q: What is your preferred model strategy?**
A: Sonnet default, Opus only for complex reasoning

**Q: Tool access philosophy?**
A: Minimal - only essential tools per agent type

---

## Key Patterns from Library

1. **Subagent Location:** `.claude/agents/{name}.md`
2. **Built-in Agents:** Explore, Plan, Bash (don't duplicate)
3. **Hook Integration:** Agents can be triggered via hooks
4. **Context Isolation:** Subagents run in separate context
5. **Delegation Pattern:** `use subagents to investigate X`

---

## Environment Configuration Patterns

### Directory Structure
```
project/
├── .claude/
│   ├── settings.json        # Team settings (git tracked)
│   ├── settings.local.json  # Personal overrides (gitignored)
│   ├── agents/              # Custom subagents
│   ├── rules/               # Path-specific rules
│   ├── skills/              # Reusable workflows
│   └── hooks/               # Automation scripts
├── .mcp.json                # MCP server configs
└── CLAUDE.md                # Project memory
```

### Configuration Files

| File | Scope | Git Tracked | Purpose |
|------|-------|-------------|---------|
| `~/.claude/settings.json` | User | N/A | Personal global settings |
| `.claude/settings.json` | Project | Yes | Team-shared settings |
| `.claude/settings.local.json` | Project | No | Local overrides |
| `~/.claude.json` | User | N/A | OAuth, theme, MCP state |
| `.mcp.json` | Project | Yes | MCP server definitions |
| `CLAUDE.md` | Project | Yes | Project context/memory |

### Hook Events

| Event | Trigger | Use Case |
|-------|---------|----------|
| PreToolUse | Before tool runs | Validation, logging |
| PostToolUse | After tool runs | Logging, notifications |
| UserPromptSubmit | User sends message | Input processing |
| SessionStart | Session begins | Environment setup |
| SessionEnd | Session ends | Cleanup |

### MCP Scopes

| Scope | Location | Visibility |
|-------|----------|------------|
| local | `.mcp.json` | Private, current project |
| project | `.mcp.json` (git) | Shared via repository |
| user | `~/.claude.json` | All your projects |

### Environment Setup Checklist

- [ ] Create `.claude/` directory structure
- [ ] Add `.claude/settings.local.json` to `.gitignore`
- [ ] Initialize `CLAUDE.md` with project context
- [ ] Configure permission rules in settings
- [ ] Set up hooks for automation needs
- [ ] Define MCP servers if external tools needed

---

## Environment Factory Lessons (2026-01-31)

### CLAUDE.md Generation Rules

**Mandatory prefix:**
```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
```

**Anti-patterns to avoid:**
- Redundant directory trees (use tables instead)
- Generic development advice
- Obvious instructions ("write tests", "use descriptive names")
- Listing every file/component
- Fabricating sections like "Tips" or "Support"

**Preferred patterns:**
- Tables for agents, hooks, components
- Concise purpose statement
- Only project-specific conventions
- Link to existing docs rather than duplicate

### Settings File Distinction

| File | Contains | Git Tracked |
|------|----------|-------------|
| `.claude/settings.json` | Permissions, env vars, mcpServers | Yes |
| `.claude/settings.local.json` | **Hooks**, personal overrides | No (gitignored) |

**Critical:** Hooks go in `settings.local.json` because they may contain machine-specific paths.

### Agent Lifecycle Pattern

```
1. CREATE    → /agent/{name}.md           (meta-agent output)
2. TEST      → Manual verification         (invoke via Task tool)
3. DEPLOY    → .claude/agents/{name}.md   (production location)
4. REGISTER  → Update CLAUDE.md           (document in Active Agents table)
```

### Hook Configuration Patterns

**SessionStart** (runs once per session):
```javascript
// Create session folder, initialize tracking files
// Location: .claude/hooks/session-init.js
```

**PostToolUse with matcher** (runs after specific tools):
```json
{
  "matcher": "Bash",
  "command": "node .claude/hooks/logger.js \"$TOOL_INPUT\" \"$TOOL_OUTPUT\" \"$EXIT_CODE\""
}
```

**Available variables:** `$TOOL_INPUT`, `$TOOL_OUTPUT`, `$EXIT_CODE`

### Session Logging Architecture

```
sessions/
└── {project-name}/           # lowercase, hyphenated
    └── {YYYY-MM-DD}/
        ├── .current_session  # tracks active session number
        └── session-{NNN}/
            ├── meta.md       # created_at, project, cwd
            └── commands.md   # timestamped command table
```

**Timeout logic:**
- Check `.current_session` mtime
- If >30 min since last activity → increment session number
- Session number resets daily (per date folder)

### Token Optimization for Generated Files

| Element | Bad | Good |
|---------|-----|------|
| Directory listing | Full ASCII tree | Table with Purpose column |
| Agent list | Prose paragraphs | Table: Name, Model, Trigger |
| Instructions | "You should always..." | Imperative: "Use X for Y" |
| Redundancy | Repeat info in multiple sections | Single source of truth |

---

## Environment Completeness Rules (2026-01-31)

### Gitignore Consistency

**Rule:** All auto-generated folders must be gitignored together.

```gitignore
# Auto-generated by hooks (must be consistent)
sessions/
github_logs/
```

**Anti-pattern:** Ignoring some but not others → drift and confusion.

### Meta-Agent vs Subagent Placement

| Type | Location | Deploy to .claude/agents/? |
|------|----------|----------------------------|
| Meta-agents (factories) | Root `/*.md` | NO |
| Task subagents | `/agent/` → `.claude/agents/` | YES (after testing) |

### Environment Validation Checklist

Before marking setup complete:
- [ ] All auto-generated folders in `.gitignore`
- [ ] `settings.local.json` gitignored
- [ ] CLAUDE.md reflects actual state
- [ ] No meta-agents in `.claude/agents/`
- [ ] MCP file exists (even if empty)
- [ ] Skills folder has content or is documented as empty

### Sync Protocol

Update in order when environment changes:
1. Source file (actual config)
2. `best_practice/` (if reusable pattern)
3. `CLAUDE.md` (never let drift from reality)

### Current Gaps Documentation

CLAUDE.md should include a "Current Gaps" section listing:
- Uninitialized components (MCP, skills)
- Planned improvements

---

## Standard Utility Agents (2026-01-31)

**Rule:** Deploy these agents by default in every new environment.

| Agent | File | Model | Trigger |
|-------|------|-------|---------|
| code-reviewer | `code-reviewer.md` | sonnet | After Edit/Write |
| test-runner | `test-runner.md` | sonnet | After code changes |
| security-scanner | `security-scanner.md` | opus | Auth/crypto code |
| doc-writer | `doc-writer.md` | sonnet | Docs requested |

**Rationale:**
- Universal utility across all project types
- Provides consistent baseline toolkit
- Easier to remove than to add later
- Templates exist in best_practice, deploy to `.claude/agents/`

**Environment factory must:**
1. Check if utility agents exist in `.claude/agents/`
2. If missing, deploy from templates
3. Update CLAUDE.md to list them

---

## Autonomous Convergence Architecture (2026-01-31)

### Core Concept
User provides task.txt + convergence-criteria.txt → System works autonomously until criteria pass.

### Governance Layer (Opus)

| Agent | Role | Trigger |
|-------|------|---------|
| orchestrator | Master controller | Task start |
| actor-critique | Quality oversight | Every 10 work units |
| learner-agent | Knowledge extraction | After critique/vote |

### Execution Layer (Sonnet/Haiku)

| Agent | Role |
|-------|------|
| task-parser | Converts free text → structured work |
| library-lookup | RAG queries (ALL agents use first) |
| convergence-evaluator | Checks pass/fail criteria |

### Evaluation Formula
```
score = (points × 0.4) + (convergence_delta × 0.4) + (qualitative × 0.2)
```

### Voting Protocol
1. Blind vote (independent)
2. Reveal (simultaneous)
3. Discuss (if split)
4. Decide (majority or escalate to user)

### Rollback Pattern
- Git checkpoint every 10 work units
- Actor-critique triggers rollback if needed
- Learner-agent extracts lessons from failure

### Learning Decay
- Initial weight: 1.0
- Decay: -0.2 per unused task
- Refresh: Reset to 1.0 when used
- Archive: When weight < 0.2

### RAG Pattern (Mandatory)
All agents must:
1. Query library-lookup before acting
2. Apply relevant patterns from library/
3. Check learnings/ for past lessons (weighted)

### Skills
- `/setup` — One-command environment setup
- `/process-task` — Start autonomous execution
- `/vote` — Multi-agent voting protocol
