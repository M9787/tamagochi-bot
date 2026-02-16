# Claude Code Documentation Library

## Overview
This library contains comprehensive structured documentation extracted from the official Claude Code documentation at `code.claude.com/docs`.

**Last Updated:** 2026-01-28
**Source:** https://code.claude.com/docs
**Total Pages:** 52

## Quick Reference

### Data Files
- **[data.csv](../data.csv)** - Complete tabular index of all documentation (52 entries)

### Category Folders

| Category | Pages | Description |
|----------|-------|-------------|
| [getting-started](getting-started/) | 2 | Installation, quickstart, first steps |
| [core-concepts](core-concepts/) | 4 | Architecture, features, workflows, best practices |
| [outside-terminal](outside-terminal/) | 8 | Web, desktop, IDE integrations, CI/CD |
| [build-with-claude-code](build-with-claude-code/) | 8 | Subagents, plugins, skills, hooks, MCP |
| [deployment](deployment/) | 8 | Enterprise, cloud providers, sandboxing |
| [administration](administration/) | 8 | Setup, IAM, security, monitoring, costs |
| [configuration](configuration/) | 6 | Settings, memory, models, keybindings |
| [reference](reference/) | 6 | CLI, hooks, plugins, troubleshooting |
| [resources](resources/) | 2 | Legal, changelog |

## Documentation Structure

### Getting Started
1. **Overview** - What Claude Code does, why developers love it
2. **Quickstart** - Installation and first session

### Core Concepts
3. **How Claude Code Works** - Agentic loop, tools, context
4. **Features Overview** - When to use each feature
5. **Common Workflows** - Step-by-step guides
6. **Best Practices** - Tips for maximum effectiveness

### Outside Terminal
7. **Claude Code on the Web** - Cloud-based sessions
8. **Desktop App** - Local/cloud with Claude desktop
9. **Chrome Integration** - Browser automation
10. **VS Code Extension** - IDE integration
11. **JetBrains IDEs** - IntelliJ, PyCharm, etc.
12. **GitHub Actions** - CI/CD automation
13. **GitLab CI/CD** - MR automation
14. **Slack Integration** - Chat-based coding

### Build with Claude Code
15. **Custom Subagents** - Specialized AI assistants
16. **Create Plugins** - Extend with skills, hooks, MCP
17. **Discover Plugins** - Marketplace browsing
18. **Extend with Skills** - Custom slash commands
19. **Output Styles** - Adapt output format
20. **Hooks Guide** - Shell command automation
21. **Run Programmatically** - Agent SDK CLI
22. **MCP Integration** - Model Context Protocol

### Deployment
23. **Enterprise Overview** - Third-party integrations
24. **Amazon Bedrock** - AWS configuration
25. **Google Vertex AI** - GCP configuration
26. **Microsoft Foundry** - Azure configuration
27. **Network Configuration** - Proxies, certificates
28. **LLM Gateway** - Gateway solutions
29. **Development Containers** - Consistent environments
30. **Sandboxing** - Filesystem/network isolation

### Administration
31. **Setup Guide** - Installation details
32. **Identity and Access Management** - Authentication, authorization
33. **Security** - Safeguards and best practices
34. **Data Usage** - Privacy policies
35. **Monitoring Usage** - OpenTelemetry setup
36. **Cost Management** - Token tracking, optimization
37. **Analytics** - Usage insights
38. **Plugin Marketplaces** - Hosting and distribution

### Configuration
39. **Settings** - Global and project settings
40. **Terminal Configuration** - Themes, notifications
41. **Model Configuration** - Model aliases, caching
42. **Memory Management** - CLAUDE.md, rules
43. **Status Line** - Custom status display
44. **Keybindings** - Custom keyboard shortcuts

### Reference
45. **Interactive Mode** - Shortcuts, commands
46. **Checkpointing** - Track and rewind edits
47. **Hooks Reference** - Complete hook documentation
48. **Plugins Reference** - Technical specifications
49. **CLI Reference** - Commands and flags
50. **Troubleshooting** - Common issues and solutions

### Resources
51. **Legal and Compliance** - Agreements, certifications
52. **Changelog** - Version history

## Key Commands Reference

| Command | Description |
|---------|-------------|
| `claude` | Start interactive REPL |
| `claude "query"` | Start REPL with initial prompt |
| `claude -p "query"` | Query via SDK, then exit |
| `claude -c` | Continue most recent conversation |
| `claude --resume` | Resume session by ID or name |
| `claude update` | Update to latest version |
| `claude mcp` | Configure MCP servers |

## Essential Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Cancel current operation |
| `Ctrl+L` | Clear screen |
| `Shift+Tab` | Cycle permission modes |
| `Ctrl+R` | Reverse search history |
| `Ctrl+O` | Toggle verbose mode |
| `Esc` | Stop Claude mid-action |
| `Esc+Esc` | Open rewind menu |

## Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| `settings.json` | `~/.claude/` | User settings |
| `settings.json` | `.claude/` | Project settings |
| `settings.local.json` | `.claude/` | Local project settings |
| `.claude.json` | `~/` | Global state |
| `.mcp.json` | Project root | MCP servers |
| `CLAUDE.md` | Project root | Project memory |

## Model Aliases

| Alias | Model | Use Case |
|-------|-------|----------|
| `default` | Claude Sonnet 4.5 | Balanced performance |
| `sonnet` | Claude Sonnet 4.5 | Fast, cost-effective |
| `opus` | Claude Opus 4.5 | Complex reasoning |
| `haiku` | Claude Haiku | Quick tasks |
| `opusplan` | Opus for planning | Architecture decisions |

## Sources

- Official Documentation: https://code.claude.com/docs
- Documentation Index: https://code.claude.com/docs/llms.txt
- GitHub Repository: https://github.com/anthropics/claude-code
