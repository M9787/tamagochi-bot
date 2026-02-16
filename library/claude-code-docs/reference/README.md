# Reference Documentation

## CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `claude` | Start interactive REPL | `claude` |
| `claude "query"` | Start with prompt | `claude "explain this project"` |
| `claude -p "query"` | Non-interactive query | `claude -p "find bugs"` |
| `claude -c` | Continue last conversation | `claude -c` |
| `claude -r "name"` | Resume by session name | `claude -r "auth-refactor"` |
| `claude update` | Update Claude Code | `claude update` |
| `claude mcp` | Manage MCP servers | `claude mcp list` |

## CLI Flags

| Flag | Description |
|------|-------------|
| `--model` | Set model (sonnet, opus, haiku) |
| `--permission-mode` | Set mode (normal, plan, acceptEdits) |
| `--allowedTools` | Auto-approve specific tools |
| `--disallowedTools` | Block specific tools |
| `--output-format` | Output format (text, json, stream-json) |
| `--append-system-prompt` | Add to system prompt |
| `--system-prompt` | Replace system prompt |
| `--max-turns` | Limit agentic turns |
| `--max-budget-usd` | Limit API spend |
| `--verbose` | Enable verbose logging |
| `--debug` | Enable debug mode |

## Interactive Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/clear` | Clear context |
| `/compact` | Summarize and compact context |
| `/cost` | Show token usage |
| `/config` | Open configuration |
| `/permissions` | Manage permissions |
| `/sandbox` | Configure sandboxing |
| `/mcp` | Manage MCP servers |
| `/memory` | Edit CLAUDE.md |
| `/resume` | Resume past session |
| `/rename` | Rename current session |
| `/rewind` | Restore checkpoint |
| `/bug` | Report an issue |
| `/doctor` | Run diagnostics |

## Keyboard Shortcuts

### General
| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Cancel/interrupt |
| `Ctrl+L` | Clear screen |
| `Ctrl+D` | Exit |
| `Esc` | Stop Claude |
| `Esc+Esc` | Open rewind menu |
| `Shift+Tab` | Cycle permission modes |

### Text Editing
| Shortcut | Action |
|----------|--------|
| `Ctrl+A` | Start of line |
| `Ctrl+E` | End of line |
| `Ctrl+W` | Delete word backward |
| `Ctrl+K` | Delete to end of line |
| `Ctrl+U` | Delete to start of line |

### History
| Shortcut | Action |
|----------|--------|
| `Ctrl+R` | Reverse search |
| `Up/Down` | Navigate history |
| `Ctrl+P/N` | Previous/next |

### Display
| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Toggle verbose |
| `Alt+T` | Toggle thinking |
| `Ctrl+G` | Open plan in editor |

## Hook Events Reference

| Event | Trigger | Use Case |
|-------|---------|----------|
| `PreToolUse` | Before tool runs | Validation, modification |
| `PostToolUse` | After tool runs | Formatting, logging |
| `PermissionRequest` | Before permission prompt | Auto-approve/deny |
| `UserPromptSubmit` | User sends message | Add context, validate |
| `Notification` | Claude needs attention | Custom notifications |
| `Stop` | Before Claude stops | Verify completion |
| `SubagentStop` | Before subagent stops | Review results |
| `SessionStart` | Session begins | Environment setup |
| `SessionEnd` | Session ends | Cleanup, reporting |
| `PreCompact` | Before compaction | Preserve context |
| `Setup` | Initial setup | Project initialization |

## Hook Input/Output

### Input (stdin JSON)
```json
{
  "hook_event_name": "PreToolUse",
  "tool_name": "Bash",
  "tool_input": {
    "command": "npm test"
  },
  "session_id": "abc123",
  "cwd": "/path/to/project"
}
```

### Output (JSON)
```json
{
  "decision": "allow",
  "reason": "Safe command",
  "updatedInput": {
    "command": "npm test --verbose"
  }
}
```

## Exit Codes
| Code | Meaning |
|------|---------|
| 0 | Allow/success |
| 1 | Error (show stderr) |
| 2 | Block action |

## Plugin Manifest Schema

```json
{
  "name": "plugin-name",
  "version": "1.0.0",
  "description": "Plugin description",
  "author": {
    "name": "Author Name",
    "email": "author@example.com"
  },
  "homepage": "https://example.com",
  "repository": "https://github.com/user/plugin",
  "license": "MIT",
  "keywords": ["keyword1", "keyword2"],
  "commands": "./commands/",
  "agents": "./agents/",
  "skills": "./skills/",
  "hooks": "./hooks/hooks.json",
  "mcpServers": "./.mcp.json"
}
```

## Troubleshooting

### Common Issues

**Installation fails:**
```bash
# Use native installer
curl -fsSL https://claude.ai/install.sh | bash
```

**Permission errors:**
```bash
# Reset permissions
/permissions
```

**Search not working:**
```bash
# Install ripgrep
brew install ripgrep  # macOS
apt install ripgrep   # Linux
```

**High memory usage:**
```bash
# Compact context
/compact

# Clear and start fresh
/clear
```

### Diagnostic Commands
```bash
claude --version        # Check version
/doctor                 # Run diagnostics
claude --debug "test"   # Debug mode
```

### Reset Configuration
```bash
rm ~/.claude.json       # Global state
rm -rf ~/.claude/       # User settings
rm -rf .claude/         # Project settings
```
