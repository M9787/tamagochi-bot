# Getting Started with Claude Code

## Overview
Claude Code is Anthropic's agentic coding tool that lives in your terminal, understands your codebase, and helps you code faster.

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

### Other Methods
- **Homebrew (macOS):** `brew install anthropic/tap/claude-code`
- **WinGet (Windows):** `winget install Anthropic.ClaudeCode`

## First Steps

### 1. Start Claude Code
```bash
claude
```

### 2. Log In
When prompted, complete authentication via browser.

### 3. Your First Query
```
> give me an overview of this codebase
```

### 4. Make Code Changes
```
> add input validation to the login function
```

### 5. Use Git
```
> create a commit for my changes
```

## Essential Commands

| Command | Description |
|---------|-------------|
| `claude` | Start interactive session |
| `claude "query"` | Start with initial prompt |
| `claude -p "query"` | Non-interactive query |
| `claude -c` | Continue last conversation |
| `claude --resume` | Resume by session name |
| `claude update` | Update Claude Code |

## Pro Tips for Beginners

1. **Be specific** - Include file names, function names, error messages
2. **Give verification targets** - Provide test cases or expected behavior
3. **Use `/clear`** - Reset context between unrelated tasks
4. **Try Plan Mode** - Press `Shift+Tab` twice for read-only exploration
5. **Ask questions** - Claude can explain code, architecture, patterns
6. **Reference files** - Use `@filename` to include file content directly

## What Claude Code Does

- **Understands your codebase** - Reads files, explores structure
- **Writes and edits code** - Creates new files, modifies existing ones
- **Runs commands** - Executes bash commands, tests, builds
- **Creates commits and PRs** - Handles Git workflows
- **Answers questions** - Explains code, debugs issues
- **Automates workflows** - CI/CD, batch operations

## Next Steps

1. [Common Workflows](../core-concepts/common-workflows.md) - Step-by-step guides
2. [Best Practices](../core-concepts/best-practices.md) - Tips for effectiveness
3. [Configure Settings](../configuration/settings.md) - Customize Claude Code
