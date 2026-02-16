---
name: start-team
description: Starts the autonomous agent team with full governance loop. Spawns orchestrator, agent-creator, actor-critique, learner-agent, and convergence-evaluator with deterministic configs.
user_invocable: true
---

# Start Team

You are launching the Tamagochi project's autonomous agent team.

## Step 1: Get Task Input

Read `.claude/task/convergence-criteria.txt` to check if criteria are defined.

Ask the user:
1. **What is the task?** (free text description of what to accomplish)
2. **What are the success criteria?** (measurable pass/fail conditions)

Write the task to `.claude/task/task.txt`.
Update `.claude/task/convergence-criteria.txt` with the user's criteria in format:
```
[ ] Criterion 1
[ ] Criterion 2
```

## Step 2: Reset Metrics & Initialize Infrastructure

### 2a: Initialize Git (if needed)
```bash
if [ ! -d ".git" ]; then
  git init
  echo -e "output/\nlogs/\n__pycache__/\n*.pyc\n.env\nLive Data - Copy.zip\nsessions/\n.claude/settings.local.json" > .gitignore
  git add -A
  git commit -m "initial: project baseline before team execution"
fi
```

### 2b: Reset all metric files
```bash
echo "0" > .claude/metrics/work-unit-counter.txt

cat > .claude/metrics/work-log.md << 'EOF'
# Work Log

| # | Timestamp | Tool | Status |
|---|-----------|------|--------|
EOF

cat > .claude/metrics/critique-log.md << 'EOF'
# Critique Log

Reviews by actor-critique agent.
EOF

cat > .claude/metrics/convergence-log.md << 'EOF'
# Convergence Log

| # | Timestamp | Passed | Total | Pct | Delta | Status |
|---|-----------|--------|-------|-----|-------|--------|
EOF

cat > .claude/metrics/checkpoints.md << 'EOF'
# Checkpoints

| # | Timestamp | Work Units | Git Commit | Review Decision |
|---|-----------|------------|------------|-----------------|
EOF
```

### 2c: Ensure learnings/ exists
```bash
mkdir -p learnings
if [ ! -f "learnings/index.md" ]; then
  cat > learnings/index.md << 'EOF'
# Learnings Index

| # | Date | Topic | Weight | Source |
|---|------|-------|--------|--------|
EOF
fi

if [ ! -f "learnings/decay.json" ]; then
  echo "{}" > learnings/decay.json
fi
```

### 2d: Ensure best_practice/ exists
```bash
if [ ! -d "best_practice" ]; then
  echo "WARNING: best_practice/ folder missing. Agent-creator will enter Learning Mode on first use."
fi
```

## Step 3: Create Team

Create the team using TeamCreate:
```
team_name: "tamagochi-team"
description: "Autonomous governance team for Tamagochi project"
```

## Step 4: Spawn Governance Agents

Spawn these 5 agents as teammates in this order:

1. **orchestrator** (opus) — master coordinator
   - Spawn prompt: "You are the orchestrator. Read .claude/task/task.txt for your task. Read .claude/task/convergence-criteria.txt for success criteria. Read .claude/agents/orchestrator.md for your full instructions. Coordinate all work, track work units, trigger checkpoints every 10 units, manage voting when needed. Message agent-creator to spawn sonnet workers as needed. Temperature 0.03, max_tokens 1500."

2. **agent-creator** (opus) — subagent factory
   - Spawn prompt: "You are the agent-creator. Read .claude/agents/agent-creator.md for your full instructions. Read best_practice/ for patterns. Wait for orchestrator to request new sonnet workers. Generate worker agents in .claude/agents/ following best_practice patterns. Temperature 0.03, max_tokens 1500."

3. **actor-critique** (opus) — quality reviewer
   - Spawn prompt: "You are the actor-critique. Read .claude/agents/actor-critique.md for your full instructions. Wait for orchestrator to trigger your review after git checkpoints. Score work, decide CONTINUE/ROLLBACK/ADJUST/VOTE_NEEDED. Log to .claude/metrics/critique-log.md. Temperature 0.03, max_tokens 1500."

4. **learner-agent** (opus) — knowledge extractor + 3rd voter
   - Spawn prompt: "You are the learner-agent. Read .claude/agents/learner-agent.md for your full instructions. Wait for actor-critique to trigger you on rollbacks or votes. Extract lessons to learnings/. Participate in 3-way voting. Read learnings/index.md before acting. Temperature 0.03, max_tokens 1500."

5. **convergence-evaluator** (sonnet) — scorekeeper
   - Spawn prompt: "You are the convergence-evaluator. Read .claude/agents/convergence-evaluator.md for your full instructions. Read .claude/task/convergence-criteria.txt. When orchestrator requests a check, test each criterion and report pass/fail percentage. Log to .claude/metrics/convergence-log.md. Temperature 0.03, max_tokens 1500."

## Step 5: Assign Initial Task

Send message to orchestrator:
"Task is ready. Read .claude/task/task.txt and .claude/task/convergence-criteria.txt. Begin execution. Message agent-creator to spawn sonnet workers as needed."

## Step 6: Confirm to User

Report:
- Team name
- 5 agents spawned with roles (4 opus + 1 sonnet)
- Git repo initialized (or already existed)
- Task loaded
- Convergence criteria count
- All metrics reset
- "Use Shift+Up/Down to navigate teammates. Ctrl+T for task list."
- "Use /stop-team to gracefully shut down."
