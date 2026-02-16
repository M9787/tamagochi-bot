# Configuration Reference

## Configuration Files

| File | Location | Purpose | Shared |
|------|----------|---------|--------|
| `settings.json` | `~/.claude/` | User settings | No |
| `settings.json` | `.claude/` | Project settings | Yes (git) |
| `settings.local.json` | `.claude/` | Local project | No |
| `.claude.json` | `~/` | Global state | No |
| `.mcp.json` | Project root | MCP servers | Yes |
| `CLAUDE.md` | Project root | Project memory | Yes |

## Settings Schema

```json
{
  "permissions": {
    "allow": ["Bash(npm test)", "Read"],
    "deny": ["Bash(rm *)"],
    "defaultMode": "normal"
  },
  "hooks": {
    "PostToolUse": [...]
  },
  "model": "sonnet",
  "theme": "dark",
  "verbose": false
}
```

## Permission Rules

### Syntax
```
ToolName(pattern)
```

### Examples
```json
{
  "allow": [
    "Read",                    // All reads
    "Bash(npm *)",             // npm commands
    "Bash(git log *)",         // git log only
    "Edit(src/**/*.ts)"        // TypeScript in src/
  ],
  "deny": [
    "Bash(rm -rf *)",          // Destructive rm
    "Write(.env*)"             // Environment files
  ]
}
```

## Environment Variables

### API Configuration
| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | API key for Claude API |
| `ANTHROPIC_BASE_URL` | Custom API endpoint |
| `CLAUDE_CODE_USE_BEDROCK` | Use Amazon Bedrock |
| `CLAUDE_CODE_USE_VERTEX` | Use Google Vertex AI |

### Model Configuration
| Variable | Description |
|----------|-------------|
| `ANTHROPIC_MODEL` | Override default model |
| `MAX_THINKING_TOKENS` | Limit extended thinking |
| `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS` | Disable beta features |

### Proxy Configuration
| Variable | Description |
|----------|-------------|
| `HTTPS_PROXY` | HTTPS proxy URL |
| `HTTP_PROXY` | HTTP proxy URL |
| `NO_PROXY` | Bypass proxy for hosts |
| `NODE_EXTRA_CA_CERTS` | Custom CA certificates |

### Telemetry
| Variable | Description |
|----------|-------------|
| `CLAUDE_CODE_ENABLE_TELEMETRY` | Enable telemetry |
| `OTEL_METRICS_EXPORTER` | Metrics exporter |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint |

## Model Aliases

```bash
# Set model for session
claude --model opus

# Set in settings
{
  "model": "sonnet"
}

# Available aliases
default  # Claude Sonnet 4.5
sonnet   # Claude Sonnet 4.5
opus     # Claude Opus 4.5
haiku    # Claude Haiku
opusplan # Opus for planning, Sonnet for execution
```

## Memory (CLAUDE.md)

### Project Memory
Create `CLAUDE.md` in project root:
```markdown
# Project: My App

## Commands
- `npm test` - Run tests
- `npm run build` - Build for production

## Code Style
- Use TypeScript strict mode
- Prefer functional components
- Use Tailwind for styling

## Architecture
- src/api/ - API routes
- src/components/ - React components
- src/lib/ - Utility functions
```

### Modular Rules
Create `.claude/rules/`:
```
.claude/rules/
├── api.md          # API-specific rules
├── testing.md      # Test conventions
└── security.md     # Security guidelines
```

Path-specific rules in frontmatter:
```markdown
---
globs: ["src/api/**/*.ts"]
---
# API Rules
Use async/await for all handlers.
```

## Sandbox Configuration

```json
{
  "sandbox": {
    "enabled": true,
    "mode": "auto-allow",
    "filesystem": {
      "allowedPaths": ["."],
      "deniedPaths": [".git", "node_modules"]
    },
    "network": {
      "allowedHosts": ["api.example.com"]
    }
  }
}
```

## Keybindings

Create `~/.claude/keybindings.json`:
```json
{
  "bindings": [
    {
      "key": "ctrl+k",
      "command": "clear",
      "context": "input"
    },
    {
      "key": "ctrl+r",
      "command": "history-search-backward",
      "context": "input"
    }
  ]
}
```

## Status Line

Create `~/.claude/statusline.sh`:
```bash
#!/bin/bash
echo '{"left": "'"$(git branch --show-current 2>/dev/null || echo 'no-git')"'", "right": "'"$(date +%H:%M)"'"}'
```

Enable in settings:
```json
{
  "statusLine": {
    "enabled": true,
    "script": "~/.claude/statusline.sh"
  }
}
```
