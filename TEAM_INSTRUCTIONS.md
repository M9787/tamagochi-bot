# Team Mode Instructions

## Quick Start

```
/start-team
```

That's it. The skill will ask you for a task description and success criteria, then spin up everything automatically.

To stop:
```
/stop-team
```

---

## What You Have (Complete & Ready)

### 5 Governance Agents

| Agent | Model | Role | File |
|-------|-------|------|------|
| **orchestrator** | opus | Central brain. Parses tasks into work units, assigns to workers, triggers checkpoints every 10 units, coordinates voting | `.claude/agents/orchestrator.md` |
| **agent-creator** | opus | Subagent factory. Spawns sonnet workers on demand. Reads `best_practice/` for patterns | `.claude/agents/agent-creator.md` |
| **actor-critique** | opus | Quality reviewer. Scores work after checkpoints. Decides: CONTINUE / ROLLBACK / ADJUST / VOTE_NEEDED | `.claude/agents/actor-critique.md` |
| **learner-agent** | opus | Knowledge extractor + 3rd voter. Logs lessons from failures/successes to `learnings/` | `.claude/agents/learner-agent.md` |
| **convergence-evaluator** | sonnet | Scorekeeper. Tests each success criterion, reports pass/fail %, detects stalls | `.claude/agents/convergence-evaluator.md` |

### 2 User Skills

| Skill | Command | What It Does |
|-------|---------|--------------|
| **start-team** | `/start-team` | Asks for task + criteria, inits git, resets metrics, spawns all 5 agents, sends task to orchestrator |
| **stop-team** | `/stop-team` | Reads final metrics, shuts down agents in reverse order, final git commit, cleans up team |

### 2 Automation Hooks (settings.local.json)

| Hook | Trigger | What It Does |
|------|---------|--------------|
| **checkpoint-tracker.sh** | After every `Edit` or `Write` | Increments work-unit counter, logs to work-log.md, prints `CHECKPOINT_NEEDED` every 10 units |
| **session-logger.sh** | After every `Bash` | Logs command + timestamp + exit code to session-commands.md |

### Metrics Tracking (auto-populated during execution)

| File | Purpose |
|------|---------|
| `.claude/metrics/work-unit-counter.txt` | Running count of file edits/writes |
| `.claude/metrics/work-log.md` | Every work unit with timestamp |
| `.claude/metrics/checkpoints.md` | Git checkpoint log |
| `.claude/metrics/critique-log.md` | Actor-critique review decisions |
| `.claude/metrics/convergence-log.md` | Pass/fail % over time |

### Task & Convergence Files

| File | Purpose |
|------|---------|
| `.claude/task/task.txt` | Your task description (written by `/start-team`) |
| `.claude/task/convergence-criteria.txt` | Success criteria checklist (`[ ]` / `[PASS]` / `[FAIL]`) |

### Learning System

| File | Purpose |
|------|---------|
| `learnings/index.md` | Registry of all lessons learned |
| `learnings/decay.json` | Weight tracking (unused lessons decay, active ones stay) |
| `learnings/{date}_{topic}.md` | Individual lesson files (created during execution) |

---

## How It Works (Execution Flow)

```
You type: /start-team
  │
  ├─ You provide: task description + success criteria
  ├─ Git repo initialized (if needed)
  ├─ All metrics reset to zero
  ├─ 5 agents spawned (4 opus + 1 sonnet)
  └─ Task sent to orchestrator
       │
       ▼
  ORCHESTRATOR reads task → breaks into work units
       │
       ├─ Messages AGENT-CREATOR: "I need a sonnet worker for X"
       │     └─ Agent-creator generates worker agent in .claude/agents/
       │        └─ Orchestrator spawns + assigns work
       │
       ├─ Hook auto-counts every Edit/Write
       │
       └─ Every 10 work units:
            │
            ├─ Git checkpoint commit
            ├─ ACTOR-CRITIQUE reviews work
            │     │
            │     ├─ CONTINUE → CONVERGENCE-EVALUATOR checks criteria
            │     │     └─ Reports X/Y criteria passed (Z%)
            │     │
            │     ├─ ADJUST → orchestrator modifies approach
            │     │
            │     ├─ ROLLBACK → git reset + LEARNER-AGENT extracts lesson
            │     │
            │     └─ VOTE_NEEDED → 3-way blind vote
            │           (orchestrator + actor-critique + learner-agent)
            │           Majority wins. Tie → asks you.
            │
            └─ Loop until all criteria [PASS]

You type: /stop-team
  │
  ├─ Reads final metrics
  ├─ Shuts down agents (reverse order)
  ├─ Final git commit
  ├─ Cleans up team resources
  └─ Reports summary to you
```

---

## Scoring Formula (Actor-Critique)

```
score = (points x 0.4) + (convergence_delta x 0.4) + (qualitative x 0.2)

points:            +1 progress toward goal, -1 regression
convergence_delta: % change in pass/fail since last review
qualitative:       GOOD / NEUTRAL / BAD (code quality, approach soundness)
```

---

## Voting Protocol

When actor-critique and orchestrator disagree:

1. **BLIND** — Each of 3 agents votes independently (orchestrator, actor-critique, learner-agent)
2. **REVEAL** — All votes shown at once
3. **DISCUSS** — If no majority, agents can change votes with justification
4. **DECIDE** — 2/3 majority wins. All different = escalated to you

---

## Learning Decay

```
Weight starts at 1.0
Each task where learning is NOT used: weight -= 0.2
If learning IS used: weight resets to 1.0
Weight < 0.2 → ARCHIVED (removed from active consideration)
```

---

## What's Missing / Needs Attention

### Must-Fix Before First Run

| Issue | Status | Impact |
|-------|--------|--------|
| **No git repo** | `.git/` doesn't exist yet | `/start-team` will init it, but verify `.gitignore` covers sensitive files (`Api key.txt` is in `archive/`) |
| **best_practice/ not at root** | Currently at `archive/best_practice/best_practice/` | Agent-creator will enter Learning Mode on first run (reads `library/` to generate new best practices). This works but costs extra tokens. **Fix:** copy or symlink to project root |
| **Hook permissions** | `.sh` files may not be executable on Windows | Run `chmod +x .claude/hooks/*.sh` before first use (or ensure git bash handles it) |

### Nice-to-Have Improvements

| Area | Current State | Improvement |
|------|---------------|-------------|
| **Worker reuse** | Agent-creator spawns new workers each time | Could cache/reuse worker agents across tasks |
| **Convergence smartness** | Tests are pattern-matched from criterion text | Could add custom test scripts for complex criteria |
| **Metrics dashboard** | Raw markdown files | Could add a Streamlit page to visualize team metrics |
| **Auto-retraining trigger** | ML retraining is manual | Orchestrator could trigger `train.py` as convergence criterion |
| **Cost tracking** | Not tracked | Could log token usage per agent per checkpoint |
| **Parallel workers** | Sequential assignment by orchestrator | Could batch-assign independent work units |

### Known Limitations

1. **Windows shell** — Hooks use bash scripts. Works in git bash / WSL, but may fail in cmd.exe or PowerShell natively
2. **Max turns** — Orchestrator has 50 turns max. Very large tasks may need multiple `/start-team` sessions
3. **No persistence across sessions** — If Claude Code session ends, team state is lost (but git checkpoints preserve code progress)
4. **Hook counting** — Every Edit/Write increments counter, including metrics files themselves. Counter may inflate slightly

---

## Directory Structure

```
.claude/
├── agents/                       # 5 governance agent definitions
│   ├── orchestrator.md
│   ├── agent-creator.md
│   ├── actor-critique.md
│   ├── learner-agent.md
│   └── convergence-evaluator.md
├── skills/                       # User-invocable commands
│   ├── start-team/SKILL.md
│   └── stop-team/SKILL.md
├── hooks/                        # PostToolUse automation
│   ├── checkpoint-tracker.sh     # WU counter (Edit/Write)
│   └── session-logger.sh         # Command logger (Bash)
├── metrics/                      # Real-time tracking
│   ├── work-unit-counter.txt
│   ├── work-log.md
│   ├── checkpoints.md
│   ├── critique-log.md
│   ├── convergence-log.md
│   └── session-commands.md
├── task/                         # Current task & criteria
│   ├── task.txt
│   └── convergence-criteria.txt
└── settings.local.json           # Hook config (gitignored)

learnings/                        # Knowledge base
├── index.md
└── decay.json

library/                          # Reference docs for agent-creator
└── claude-code-docs/
```

---

## Example Usage

```
> /start-team

Claude: What is the task?
You: "Refactor the ML pipeline to add walk-forward validation with 3-month rolling windows"

Claude: What are the success criteria?
You:
1. train.py supports --validation-mode walk-forward
2. At least 3 rolling windows evaluated
3. All existing tests still pass
4. Results saved to model_training/results/walk_forward/

Claude: Team "tamagochi-team" created.
  - orchestrator (opus) ✓
  - agent-creator (opus) ✓
  - actor-critique (opus) ✓
  - learner-agent (opus) ✓
  - convergence-evaluator (sonnet) ✓
  Git initialized. 4 convergence criteria loaded. All metrics reset.

  Use Shift+Up/Down to navigate teammates.
  Use /stop-team to shut down.
```

Then the team works autonomously. You'll see teammate messages as they progress, checkpoint, review, and converge.
