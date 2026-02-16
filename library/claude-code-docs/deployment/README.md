# Deployment Guide

## Deployment Options

| Option | Best For | Data Residency |
|--------|----------|----------------|
| Claude API | Direct access, simplest setup | Anthropic servers |
| Amazon Bedrock | AWS infrastructure | Your AWS account |
| Google Vertex AI | GCP infrastructure | Your GCP project |
| Microsoft Foundry | Azure infrastructure | Your Azure subscription |
| LLM Gateway | Centralized proxy | Custom |

## Amazon Bedrock Setup

### Prerequisites
- AWS account with Bedrock access
- IAM role with Bedrock permissions

### Configuration
```bash
export CLAUDE_CODE_USE_BEDROCK=1
export AWS_REGION=us-west-2
export ANTHROPIC_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

### IAM Policy
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.*"
    }
  ]
}
```

## Google Vertex AI Setup

### Prerequisites
- GCP project with Vertex AI API enabled
- Service account with Vertex AI permissions

### Configuration
```bash
export CLAUDE_CODE_USE_VERTEX=1
export CLOUD_ML_REGION=us-east5
export ANTHROPIC_VERTEX_PROJECT_ID=your-project-id
```

### IAM Roles
- `roles/aiplatform.user`
- `roles/serviceusage.serviceUsageConsumer`

## Microsoft Foundry Setup

### Configuration
```bash
export AZURE_FOUNDRY_RESOURCE_NAME=your-resource
export AZURE_API_KEY=your-key
# Or use Entra ID
export AZURE_USE_ENTRA_ID=1
```

## Network Configuration

### Proxy Setup
```bash
export HTTPS_PROXY=https://proxy.example.com:8080
export HTTP_PROXY=http://proxy.example.com:8080
export NO_PROXY="localhost,127.0.0.1,.internal.com"
```

### Custom CA Certificates
```bash
export NODE_EXTRA_CA_CERTS=/path/to/ca-cert.pem
```

### mTLS Authentication
```bash
export CLAUDE_CODE_CLIENT_CERT=/path/to/client-cert.pem
export CLAUDE_CODE_CLIENT_KEY=/path/to/client-key.pem
```

### Required URLs
Allow these URLs through your firewall:
- `api.anthropic.com` - Claude API
- `claude.ai` - WebFetch safeguards
- `statsig.anthropic.com` - Telemetry
- `sentry.io` - Error reporting

## LLM Gateway

### Gateway Requirements
Must support one of:
1. Anthropic Messages: `/v1/messages`
2. Bedrock InvokeModel: `/invoke`
3. Vertex rawPredict: `:rawPredict`

### LiteLLM Configuration
```bash
export ANTHROPIC_BASE_URL=https://litellm-server:4000
export ANTHROPIC_AUTH_TOKEN=sk-litellm-key
```

## Development Containers

### Quick Start
1. Install VS Code + Remote Containers extension
2. Clone Claude Code reference implementation
3. Open in VS Code
4. Click "Reopen in Container"

### Security Features
- Firewall restricting outbound connections
- Whitelisted domains only
- Isolated development environment
- Safe for `--dangerously-skip-permissions`

## Sandboxing

### Enable Sandboxing
```bash
/sandbox
```

### Sandbox Modes
| Mode | Behavior |
|------|----------|
| Auto-allow | Sandboxed commands run without prompting |
| Regular | All commands go through permission flow |

### Prerequisites
- **macOS**: Built-in Seatbelt framework
- **Linux/WSL2**: `apt install bubblewrap socat`

### Configuration
```json
{
  "sandbox": {
    "enabled": true,
    "mode": "auto-allow",
    "filesystem": {
      "allowedPaths": ["."],
      "deniedPaths": [".git"]
    },
    "network": {
      "allowedHosts": ["api.example.com"]
    }
  }
}
```

### Security Benefits
- Filesystem isolation (read/write restrictions)
- Network isolation (domain allowlisting)
- OS-level enforcement
- Protection against prompt injection
