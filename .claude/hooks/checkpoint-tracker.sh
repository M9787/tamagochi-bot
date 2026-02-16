#!/bin/bash
# Tracks Edit/Write tool calls and increments work unit counter
# Triggered by PostToolUse on Edit and Write

COUNTER_FILE="$(dirname "$0")/../metrics/work-unit-counter.txt"
LOG_FILE="$(dirname "$0")/../metrics/work-log.md"

# Initialize counter if missing
if [ ! -f "$COUNTER_FILE" ]; then
    echo "0" > "$COUNTER_FILE"
fi

# Initialize log if missing
if [ ! -f "$LOG_FILE" ]; then
    echo "# Work Log" > "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    echo "| # | Timestamp | Tool | Status |" >> "$LOG_FILE"
    echo "|---|-----------|------|--------|" >> "$LOG_FILE"
fi

# Increment counter
COUNT=$(cat "$COUNTER_FILE")
COUNT=$((COUNT + 1))
echo "$COUNT" > "$COUNTER_FILE"

# Log entry
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "| $COUNT | $TIMESTAMP | file-change | done |" >> "$LOG_FILE"

# Check if checkpoint needed (every 10)
if [ $((COUNT % 10)) -eq 0 ]; then
    echo "CHECKPOINT_NEEDED: Work unit $COUNT reached. Run git checkpoint." >&2
fi
