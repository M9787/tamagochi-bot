---
name: stop-team
description: Gracefully shuts down the agent team, reports final metrics, and cleans up team resources.
user_invocable: true
---

# Stop Team

You are shutting down the Tamagochi project's autonomous agent team.

## Step 1: Read Final Metrics

Read these files and collect stats:
- `.claude/metrics/work-unit-counter.txt` — total work units
- `.claude/metrics/checkpoints.md` — checkpoint count
- `.claude/metrics/critique-log.md` — review count and last decision
- `.claude/metrics/convergence-log.md` — latest convergence %
- `.claude/task/convergence-criteria.txt` — pass/fail summary

## Step 2: Shutdown All Agents

Send shutdown_request to each teammate in reverse order:
1. **convergence-evaluator**
2. **learner-agent**
3. **actor-critique**
4. **agent-creator**
5. **orchestrator**

Wait for each to confirm shutdown before proceeding to next.

## Step 3: Final Git Checkpoint

```bash
git add -A
git commit -m "team-shutdown: final state after team execution"
```

## Step 4: Clean Up Team

Run TeamDelete to remove team resources.

## Step 5: Report to User

Report:
- Total work units completed
- Total checkpoints
- Total reviews (with CONTINUE/ROLLBACK/ADJUST breakdown)
- Final convergence % and criteria status
- Number of learnings generated
- "Team shut down. All metrics preserved in .claude/metrics/."
