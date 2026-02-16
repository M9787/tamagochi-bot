#!/bin/bash
# Logs all Bash tool calls with timestamp
# Triggered by PostToolUse on Bash

LOG_DIR="$(dirname "$0")/../metrics"
LOG_FILE="$LOG_DIR/session-commands.md"

# Initialize if missing
if [ ! -f "$LOG_FILE" ]; then
    echo "# Session Commands" > "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    echo "| Timestamp | Command | Exit Code |" >> "$LOG_FILE"
    echo "|-----------|---------|-----------|" >> "$LOG_FILE"
fi

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
# $1 = tool input (command), $2 = exit code
CMD=$(echo "$1" | head -c 100 | tr '|' '/')
EXIT_CODE="${2:-0}"

echo "| $TIMESTAMP | $CMD | $EXIT_CODE |" >> "$LOG_FILE"
