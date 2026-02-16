# Administration Guide

## Installation

### Native Installation (Recommended)

**macOS/Linux/WSL:**
```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**Windows PowerShell:**
```powershell
irm https://claude.ai/install.ps1 | iex
```

### Verify Installation
```bash
claude --version
```

### Update
```bash
claude update
```

## Authentication

### For Individuals
1. Run `claude`
2. Complete browser OAuth flow
3. Credentials stored in `~/.config/claude-code/auth.json`

### For Teams/Organizations
1. Configure Claude for Teams or Enterprise in Console
2. Users authenticate via SSO
3. Centralized cost and usage tracking

### Cloud Provider Authentication
- **Bedrock**: AWS IAM credentials
- **Vertex AI**: GCP service account
- **Foundry**: Azure API key or Entra ID

## Permission System

### Permission Modes
| Mode | Description |
|------|-------------|
| `normal` | Prompt for each action |
| `plan` | Read-only exploration |
| `acceptEdits` | Auto-accept file edits |

### Permission Rules
```json
{
  "permissions": {
    "allow": [
      "Read",
      "Bash(npm test)",
      "Bash(git *)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Write(.env*)"
    ],
    "defaultMode": "normal"
  }
}
```

### Managed Settings
Deploy organization-wide settings via:
- **macOS**: `/Library/Application Support/ClaudeCode/managed-settings.json`
- **Linux/WSL**: `/etc/claude-code/managed-settings.json`
- **Windows**: `C:\Program Files\ClaudeCode\managed-settings.json`

## Security

### Built-in Protections
- Permission prompts before actions
- Sandboxed execution (optional)
- Checkpoints for easy reversion
- Git integration for change tracking

### Prompt Injection Protection
- Core protections in system prompt
- Privacy safeguards for sensitive files
- MCP server security considerations

### Best Practices
1. Review `allow` rules carefully
2. Use sandboxing for untrusted code
3. Enable checkpoints
4. Review changes before committing
5. Avoid `--dangerously-skip-permissions` without sandbox

## Monitoring

### Enable OpenTelemetry
```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_METRICS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317
```

### Available Metrics
| Metric | Description |
|--------|-------------|
| `claude_code.session.count` | Sessions started |
| `claude_code.lines_of_code.count` | Lines modified |
| `claude_code.cost.usage` | Cost in USD |
| `claude_code.token.usage` | Tokens used |
| `claude_code.active_time.total` | Active time |

## Cost Management

### Track Costs
```
/cost
```

### Reduce Token Usage
1. `/clear` between unrelated tasks
2. Use Sonnet for most tasks (cheaper than Opus)
3. Disable unused MCP servers
4. Install code intelligence plugins
5. Use subagents for verbose operations
6. Adjust extended thinking budget

### Rate Limits (Recommended)
| Team Size | TPM/User | RPM/User |
|-----------|----------|----------|
| 1-5 | 200k-300k | 5-7 |
| 5-20 | 100k-150k | 2.5-3.5 |
| 20-50 | 50k-75k | 1.25-1.75 |
| 50-100 | 25k-35k | 0.62-0.87 |
| 100+ | 10k-20k | 0.25-0.5 |

## Analytics

### Access Dashboard
Navigate to: `console.anthropic.com/claude-code`

### Available Metrics
- Lines of code accepted
- Suggestion accept rate
- Activity (users, sessions per day)
- Spend (cost per day)
- Team insights

### Required Roles
Primary Owner, Owner, Billing, Admin, Developer

## Plugin Marketplaces

### Create Marketplace
Create `.claude-plugin/marketplace.json`:
```json
{
  "name": "company-tools",
  "owner": {
    "name": "DevTools Team",
    "email": "devtools@example.com"
  },
  "plugins": [
    {
      "name": "code-formatter",
      "source": "./plugins/formatter",
      "description": "Auto code formatting"
    }
  ]
}
```

### Host Options
- GitHub repository (recommended)
- GitLab, Bitbucket
- Local path for development

### Managed Restrictions
```json
{
  "strictKnownMarketplaces": [
    "https://github.com/company/marketplace"
  ]
}
```

## Data Usage

### Training Policy
- Claude Code **does not** train on your code
- Conversations not used for model training
- Data retention for operational purposes only

### Telemetry
- Opt-in only
- No code content in metrics
- User prompt content redacted by default
