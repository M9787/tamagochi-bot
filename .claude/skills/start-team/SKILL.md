---
name: start-team
description: Starts the autonomous agent team with governance loop. Spawns orchestrator, actor-critique, learner-agent, convergence-evaluator. Agent-creator is lazy-spawned only when needed.
user_invocable: true
---

# Start Team

Launch the Tamagochi project's autonomous agent team.

## Step 1: Get Task Input

Read `.claude/task/convergence-criteria.txt` to check if criteria exist.

Ask the user:
1. **What is the task?** (free text)
2. **What are the success criteria?** (measurable pass/fail)

Write task to `.claude/task/task.txt`.
Update `.claude/task/convergence-criteria.txt`:
```
[ ] Criterion 1
[ ] Criterion 2
```

## Step 2: Reset Metrics & Initialize

### 2a: Git baseline (if needed)
```bash
if [ ! -d ".git" ]; then
  git init
  echo -e "output/\nlogs/\n__pycache__/\n*.pyc\n.env\nLive Data - Copy.zip\nsessions/\n.claude/settings.local.json" > .gitignore
  git add -A
  git commit -m "initial: project baseline before team execution"
fi
```

### 2b: Reset metrics
```bash
echo "0" > .claude/metrics/work-unit-counter.txt

cat > .claude/metrics/work-log.md << 'EOF'
# Work Log

| # | Timestamp | Description | Status |
|---|-----------|-------------|--------|
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

### 2c: Ensure learnings infrastructure
```bash
mkdir -p learnings
if [ ! -f "learnings/fails-to-avoid.md" ]; then
  cat > learnings/fails-to-avoid.md << 'EOF'
# Fails to Avoid

## Index
| # | Pattern | Category | Date |
|---|---------|----------|------|
EOF
fi
```

### 2d: Verify best_practice/ exists
```bash
if [ ! -d "best_practice" ]; then
  echo "WARNING: best_practice/ missing. Agent-creator will need patterns on first use."
fi
```

## Step 3: Create Team

```
TeamCreate:
  team_name: "tamagochi-team"
  description: "Autonomous governance team for Tamagochi project"
```

## Step 4: Spawn Core Agents (4 agents — NOT 5)

Spawn in this order. **Agent-creator is NOT spawned at startup** (lazy init — only when needed).

1. **orchestrator** (opus) — master coordinator
   - Prompt: "You are the orchestrator. Read .claude/agents/orchestrator.md for your instructions. Read .claude/task/task.txt for the task. Read .claude/task/convergence-criteria.txt for success criteria. Read learnings/fails-to-avoid.md for known failure patterns. Begin execution: parse task into work units, spawn sonnet workers as needed (create worker .md files directly or message agent-creator if you need a specialized worker). Track WUs, checkpoint every 10, trigger actor-critique for review. On ADJUST: forward fixes to worker directly. max_tokens=1000, temperature=0.04."

2. **actor-critique** (opus) — quality reviewer + direct fix router
   - Prompt: "You are the actor-critique. Read .claude/agents/actor-critique.md for your instructions. Read learnings/fails-to-avoid.md for known failure patterns. Wait for orchestrator to trigger review after checkpoints. On ADJUST: send SELF-CORRECT message directly to the worker, then notify orchestrator. On unknown errors: trigger learner-agent. max_tokens=1500, temperature=0.03."

3. **learner-agent** (opus) — failure pattern extractor
   - Prompt: "You are the learner-agent. Read .claude/agents/learner-agent.md for your instructions. Read learnings/fails-to-avoid.md for existing patterns. You are triggered by ANY agent that hits an error. Analyze the failure, extract the pattern, write to learnings/fails-to-avoid.md. You run in parallel — do not block. max_tokens=1000, temperature=0.3."

4. **convergence-evaluator** (sonnet) — scorekeeper
   - Prompt: "You are the convergence-evaluator. Read .claude/agents/convergence-evaluator.md for your instructions. Read .claude/task/convergence-criteria.txt. When orchestrator requests a check, test each criterion and report pass/fail. Log to .claude/metrics/convergence-log.md. max_tokens=1500, temperature=0.03."

## Step 5: Assign Task

Send to orchestrator:
"Task is ready. Begin execution. Read your instructions, the task file, convergence criteria, and fails-to-avoid. Spawn sonnet workers as needed — you can create simple worker .md files directly or request agent-creator for complex cases."

## Step 6: Confirm to User

Report:
- Team name
- 4 agents spawned (3 opus + 1 sonnet). Agent-creator available on-demand.
- Git baseline status
- Task loaded
- Convergence criteria count
- Known failure patterns: {count from fails-to-avoid.md}
- "Use /stop-team to gracefully shut down."
