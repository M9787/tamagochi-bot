# Outside Terminal Integrations

## Claude Code on the Web

Run Claude Code sessions asynchronously on cloud infrastructure.

### Access
- URL: `claude.ai/code`
- Requirements: Pro, Max, Team, or Enterprise plan

### Features
- Run tasks in parallel
- Review changes with diff view
- Create PRs directly from sessions
- Teleport sessions to local terminal

### Moving Tasks

**Terminal to Web:**
```bash
claude --remote "Fix the login bug"
```

**Web to Terminal:**
```bash
claude --teleport
```

## Desktop App

### Features
- Visual session management
- Inline diff viewing
- Git worktree support
- Local and cloud execution

### Installation
Download from: `claude.ai/download`

## Chrome Integration

Connect Claude Code to your browser for web automation.

### Setup
1. Install Chrome extension
2. Start Claude Code with `claude --chrome`
3. Authenticate if prompted

### Capabilities
- Test web applications
- Debug with console logs
- Automate form filling
- Extract data from web pages
- Record demo GIFs

### Example
```
> test the login form at http://localhost:3000/login
```

## VS Code Extension

### Installation
1. Open VS Code
2. Extensions → Search "Claude Code"
3. Install

### Features
- Inline diff viewing
- @-mentions for files
- Plan review and approval
- Keyboard shortcuts
- MCP integration

### Key Shortcuts
| Shortcut | Action |
|----------|--------|
| `Cmd+Shift+P` | Command palette |
| `Cmd+K` | Open Claude |

## JetBrains IDEs

### Supported IDEs
IntelliJ, PyCharm, WebStorm, GoLand, PhpStorm, RubyMine, CLion, Rider, Android Studio

### Installation
Settings → Plugins → Marketplace → "Claude Code"

### Usage
- From IDE: Tools → Claude Code
- From Terminal: Claude auto-detects IDE

## GitHub Actions

Automate code reviews, PR creation, and issue resolution.

### Quick Setup
1. Install Claude GitHub app
2. Add `ANTHROPIC_API_KEY` secret
3. Create workflow file

### Workflow Example
```yaml
name: Claude Code
on:
  issue_comment:
    types: [created]
  pull_request_review_comment:
    types: [created]

jobs:
  claude:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          github_token: ${{ secrets.GITHUB_TOKEN }}
```

### Triggers
- `@claude` in issue/PR comments
- Manual workflow dispatch
- PR events

## GitLab CI/CD

### Quick Setup
1. Add `ANTHROPIC_API_KEY` as CI/CD variable
2. Add job to `.gitlab-ci.yml`

### Example Job
```yaml
claude:
  stage: ai
  image: node:24-alpine3.21
  script:
    - curl -fsSL https://claude.ai/install.sh | bash
    - claude -p "Review this MR" --permission-mode acceptEdits
```

### Triggers
- `@claude` mentions
- MR events
- Manual pipeline runs

## Slack Integration

### Setup
1. Install Claude app from Slack Marketplace
2. Connect Claude account in App Home
3. Configure GitHub repositories

### Usage
```
@claude investigate the login bug in the auth module
```

### Routing Modes
| Mode | Behavior |
|------|----------|
| Code only | All @mentions → Claude Code |
| Code + Chat | Smart routing based on intent |

### Message Actions
- View Session
- Create PR
- Retry as Code
- Change Repo

### Limitations
- GitHub repositories only
- One PR per session
- Channel only (not DMs)
