# CLAUDE.md Update Log

## 2026-03-05 (Session 2)

**Changes detected**: System consolidation — structured JSONL logging, dashboard containerization, data alignment, GCE deploy
**Files updated**: CLAUDE.md (commands section), docs/architecture.md (file tree), docs/deployment.md (4 services + logging)
**Summary**: Added core/structured_log.py, dashboard/Dockerfile, requirements-dashboard.txt. Docker compose now runs 4 containers. All services write JSONL logs with rotation. Dashboard reads from data service volume. Backfill reuses data service klines.

## 2026-03-05

**Changes detected**: Major restructure — CLAUDE.md slimmed from 511 to 87 lines
**Files updated**: CLAUDE.md, MEMORY.md, docs/architecture.md, docs/ml-pipeline.md, docs/trading-bot.md, docs/telegram-bot.md, docs/deployment.md, docs/pipeline-alignment.md
**Summary**: Redistributed all secondary information from CLAUDE.md into 6 docs/ reference files. CLAUDE.md now contains only core identity, algorithm, signal theory, constraints, commands, and reference table. Context savings: 83% (34KB -> 6KB loaded per session).
