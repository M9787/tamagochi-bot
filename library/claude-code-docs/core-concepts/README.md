# Core Concepts

## The Agentic Loop

Claude Code operates in an agentic loop:
1. **Receive input** from user
2. **Analyze** the request and context
3. **Use tools** (read files, run commands, edit code)
4. **Return results** to user
5. **Repeat** until task complete

## Available Models

| Model | Alias | Best For |
|-------|-------|----------|
| Claude Sonnet 4.5 | `sonnet`, `default` | Most tasks, balanced speed/quality |
| Claude Opus 4.5 | `opus` | Complex reasoning, architecture |
| Claude Haiku | `haiku` | Quick, simple tasks |

## Built-in Tools

### File Operations
- **Read** - Read file contents
- **Write** - Create new files
- **Edit** - Modify existing files
- **Glob** - Find files by pattern
- **Grep** - Search file contents

### Execution
- **Bash** - Run shell commands
- **NotebookEdit** - Modify Jupyter notebooks

### Communication
- **Task** - Spawn subagents
- **WebFetch** - Fetch web content
- **WebSearch** - Search the internet
- **AskUserQuestion** - Get user input

## Context Management

### Context Window
- Claude has a limited context window (~200K tokens)
- Automatic compaction when context fills up
- Use `/clear` between unrelated tasks
- Use `/compact` to manually summarize

### Session Persistence
- Conversations saved locally
- Resume with `claude --continue` or `--resume`
- Name sessions with `/rename` for easy finding

## Permission Modes

| Mode | Description |
|------|-------------|
| **Normal** | Prompts for each action |
| **Accept Edits** | Auto-accepts file edits |
| **Plan** | Read-only exploration |

Toggle with `Shift+Tab`

## Features Comparison

### When to Use Each Feature

| Feature | Purpose | Example |
|---------|---------|---------|
| **CLAUDE.md** | Project context, conventions | "Always use TypeScript" |
| **Skills** | Reusable workflows, domain knowledge | "/deploy" command |
| **Subagents** | Isolated tasks, specialized roles | Security reviewer |
| **Hooks** | Guaranteed actions, validation | Auto-lint on edit |
| **MCP** | External tool integration | Sentry, GitHub, Notion |
| **Plugins** | Bundled extensions | Code intelligence |

## Best Practices Summary

1. **Give verification targets** - Tests, expected output, screenshots
2. **Explore before implementing** - Use Plan Mode first
3. **Be specific** - Include file names, function names
4. **Configure permissions** - Set up `/permissions` or `/sandbox`
5. **Use subagents for research** - Keep main context clean
6. **Course-correct early** - Stop and redirect quickly
7. **Manage context** - `/clear` between tasks
8. **Name sessions** - Use `/rename` for tracking
