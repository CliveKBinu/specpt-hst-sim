# OpenCode Go Model Availability

Tested 2026-06-12 via `hermes chat -q "Just say DONE" --provider opencode-go --model <model> --quiet`

| Model | Status | Notes |
|---|---|---|
| `opencode-go/deepseek-v4-flash` | ✅ Works | Fast, cheap. Use for orchestrator, runner, memory |
| `opencode-go/deepseek-v4-pro` | ✅ Works | More capable, slower. Use for analyst, experimenter |
| `opencode-go/minimax-m3` | ❌ 404 | Not available on opencode-go |
| `opencode-go/minimax-m2.5` | ❌ 404 | Not available on opencode-go |

## Assignment

| Role | Model |
|---|---|
| Orchestrator (default) | `deepseek-v4-flash` |
| Analyst (delegate) | `deepseek-v4-pro` |
| Experimenter (delegate) | `deepseek-v4-pro` |
| Runner (delegate) | `deepseek-v4-flash` |
| Memory (delegate) | `deepseek-v4-flash` |
