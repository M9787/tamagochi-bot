---
name: stop-team
description: Gracefully shuts down the agent team, reports final metrics, and cleans up team resources.
user_invocable: true
---

# Stop Team

Shut down the Tamagochi project's autonomous agent team.

## Step 1: Read Final Metrics

Read and collect stats from:
- `.claude/metrics/work-unit-counter.txt` — total work units
- `.claude/metrics/checkpoints.md` — checkpoint count
- `.claude/metrics/critique-log.md` — review count and last decision
- `.claude/metrics/convergence-log.md` — latest convergence %
- `.claude/task/convergence-criteria.txt` — pass/fail summary
- `learnings/fails-to-avoid.md` — new failure patterns learned

## Step 2: Shutdown Agents

Send shutdown_request to each teammate. Order: workers first, then governance.

If agent-creator was spawned during this run, shut it down too.

1. **Any active workers** (sonnet)
2. **convergence-evaluator** (sonnet)
3. **learner-agent** (opus)
4. **actor-critique** (opus)
5. **orchestrator** (opus)

Wait for each to confirm before proceeding to next.

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
- Total reviews (CONTINUE/ADJUST/ROLLBACK breakdown)
- Final convergence % and criteria status
- New failure patterns learned (count from fails-to-avoid.md)
- "Team shut down. Metrics in .claude/metrics/. Learnings in learnings/fails-to-avoid.md."
