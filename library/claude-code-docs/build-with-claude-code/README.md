# Build with Claude Code

## Subagents

Create specialized AI assistants for specific tasks.

### Create a Subagent

Create `.claude/agents/security-reviewer.md`:
```markdown
---
name: security-reviewer
description: Reviews code for security vulnerabilities
tools: Read, Grep, Glob, Bash
model: opus
---
You are a senior security engineer. Review code for:
- Injection vulnerabilities (SQL, XSS, command injection)
- Authentication and authorization flaws
- Secrets or credentials in code
- Insecure data handling

Provide specific line references and suggested fixes.
```

### Built-in Subagents
- **Explore** - Codebase exploration
- **Plan** - Implementation planning
- **Bash** - Command execution

## Skills

Create reusable workflows and domain knowledge.

### Create a Skill

Create `.claude/skills/deploy/SKILL.md`:
```markdown
---
name: deploy
description: Deploy to production
disable-model-invocation: true
---
Deploy the application to production:

1. Run tests: `npm test`
2. Build: `npm run build`
3. Deploy: `./deploy.sh production`
4. Verify: `curl https://app.example.com/health`
```

Invoke with `/deploy` or let Claude use automatically.

## Hooks

Run shell commands at specific points in Claude's workflow.

### Hook Events
- **PreToolUse** - Before tool execution
- **PostToolUse** - After tool execution
- **UserPromptSubmit** - When user sends message
- **Notification** - When Claude needs attention
- **Stop** - Before Claude stops
- **SessionStart** - When session begins
- **SessionEnd** - When session ends

### Example: Auto-lint on Edit

`.claude/settings.json`:
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "eslint --fix \"$TOOL_INPUT_FILE_PATH\""
          }
        ]
      }
    ]
  }
}
```

## MCP (Model Context Protocol)

Connect Claude to external tools and services.

### Add MCP Server

```bash
# HTTP server (recommended)
claude mcp add --transport http github https://api.githubcopilot.com/mcp/

# Stdio server
claude mcp add --transport stdio airtable -- npx -y airtable-mcp-server
```

### Popular MCP Servers
- GitHub - PR reviews, issues
- Sentry - Error monitoring
- Notion - Documentation
- PostgreSQL - Database queries
- Slack - Team communication

### Manage Servers
```bash
claude mcp list        # List configured servers
claude mcp get <name>  # Get server details
claude mcp remove <name>  # Remove server
/mcp                   # Interactive management
```

## Plugins

Bundle skills, hooks, subagents, and MCP servers.

### Install Plugins

```bash
/plugin                           # Browse marketplace
claude plugin install <plugin>   # Install plugin
claude plugin list               # List installed
```

### Create a Plugin

Create `.claude-plugin/plugin.json`:
```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "description": "My custom plugin"
}
```

Add components:
- `commands/` - Skill markdown files
- `agents/` - Subagent definitions
- `hooks/hooks.json` - Hook configuration
- `.mcp.json` - MCP server definitions

## Headless Mode (Agent SDK)

Run Claude programmatically.

```bash
# Simple query
claude -p "What does this project do?"

# JSON output
claude -p "Summarize this project" --output-format json

# With permissions
claude -p "Run tests and fix failures" --allowedTools "Bash,Read,Edit"

# Continue conversation
claude -p "Now add logging" --continue
```

## Output Styles

Adapt Claude's output format.

Built-in styles:
- **default** - Standard coding assistant
- **concise** - Minimal output
- **verbose** - Detailed explanations
- **educational** - Teaching mode

Change with `/config` or `--output-style`.
