# Kestrel

A compact, tool-using LLM agent in about 3,000 lines of Python. It talks to any OpenAI-compatible
endpoint over plain `httpx` (no agent framework) and is built around one idea: **make the agent loop
boring, inspectable and recoverable**.

```
you> read notes.txt and tell me what the code word is
-> read_file({"path": "notes.txt"})
<- read_file [ok]      1	The code word is PERIDOT-42.
kestrel> The code word is PERIDOT-42.
```

## What it demonstrates

| Capability | What is implemented | Where |
|---|---|---|
| **Agent loop / tool use** | ReAct-style loop with a step budget, streaming, bounded retry of empty replies, and refusal to execute tool calls from truncated output | `agent.py` |
| **Tool registry & validation** | One pydantic model produces both the JSON Schema shown to the model and the strict validator; unknown/mistyped arguments become readable errors the model can fix; timeouts, output truncation, idempotency flags | `tools/registry.py` |
| **Structured state & recoverability** | Append-only SQLite message log, optimistic concurrency (`expected_head`, versioned key/value), crash recovery that re-runs idempotent tools and marks the rest "outcome unknown" | `store.py`, `agent.py` |
| **Retrieval (RAG)** | Hybrid search over past conversations: dense embeddings + BM25 fused with Reciprocal Rank Fusion; offline hashing embedder fallback; CJK-aware tokenizer | `memory/episodic.py`, `memory/bm25.py` |
| **Context management** | Token-budgeted compaction that summarises old turns *non-destructively* (a summary row hides them from the prompt; the log is never edited) and never splits a tool call from its result | `context.py` |
| **Long-term memory** | Curated Markdown facts file with a character quota, deduplication, model-driven add/remove operations after compaction, and an explicit `remember`/`forget` tool | `memory/longterm.py` |
| **Skills** | Folders with a `SKILL.md`; only a one-line index is in the prompt, full instructions load on demand; dependency checks mark skills unavailable; layered override order | `skills.py` |
| **MCP client** | Connects to stdio MCP servers, registers their tools with schema validation, labels their output as untrusted, isolates per-server failures | `mcp_client.py` |
| **Multi-agent delegation** | Parent spawns child agents with a restricted tool profile, bounded concurrency, sync or background mode, restart recovery, exactly-once result delivery | `subagents.py` |
| **Guardrails** | Approval modes (`ask` / `auto` / `readonly`), workspace path sandbox (symlink-safe), shell deny-list, credential scrubbing of subprocess environments, untrusted-content labelling | `tools/guardrails.py`, `tools/builtin.py` |
| **Reliability engineering** | Exponential backoff with jitter, `Retry-After`, no retry after tokens were streamed, tool timeouts that kill the whole process group | `llm/client.py` |
| **Observability** | Every model call, tool call and recovery action is an event; `kestrel trace` prints a timeline with token totals and optional cost estimate | `tracing.py` |
| **Web interface** | Local chat UI with streaming, collapsible tool-call cards, in-page approve/deny for risky actions, session list, trace view, memory editor and search, tool/skill/MCP overview, sub-agent jobs; token-protected | `web/` |
| **Telegram channel** | Talk to the same agent from a Telegram private chat: long polling over httpx, deny-by-default allow-list, one session per chat shared with the web UI, Allow/Deny buttons for risky tools, persisted offsets, token redaction | `channels/telegram.py` |
| **Evaluation** | Declarative eval suites (`contains`, `tool_called`, `max_steps`, `file_contains`, ...), isolated workspace per case, pass-rate / steps / tokens / latency report | `evals.py` |

## Quick start

Requires Python 3.11+ and a copy of this repository (`git clone`, then `cd kestrel`).

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env              # then set KESTREL_API_KEY (and optionally the embedding key)
```

**Windows (PowerShell)**

```powershell
py -3 -m venv venv                # or: python -m venv venv
venv\Scripts\Activate.ps1         # if scripts are blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -e ".[dev]"
Copy-Item .env.example .env       # then set KESTREL_API_KEY (and optionally the embedding key)
```

**Windows (cmd.exe)**: `py -3 -m venv venv`, `venv\Scripts\activate.bat`, `pip install -e ".[dev]"`, `copy .env.example .env`.

Then, on any platform (run these **from the repository folder**, because `.env` is read from the current directory):

```bash
kestrel info                      # shows the effective configuration
kestrel chat                      # terminal
kestrel web                       # browser UI (prints a link with your access token)
```

No API key yet? Run the whole loop against a scripted model:

```bash
python examples/offline_demo.py
```

Where your data lives: `~/.kestrel` on macOS/Linux, `%USERPROFILE%\.kestrel` on Windows (override with `KESTREL_HOME`).

### Commands

| Command | Purpose |
|---|---|
| `kestrel telegram` | Run only the Telegram channel |
| `kestrel web [--port 8765] [--approval ...] [--mcp FILE]` | Local web interface (see below) |
| `kestrel chat [--session ID] [--approval ask\|auto\|readonly] [--mcp FILE]` | Interactive chat with streaming. `/new`, `/trace`, `/help`, `/exit` |
| `kestrel run "prompt"` | One-shot run; exit code 1 if the run did not complete |
| `kestrel sessions [--all]` | List sessions (`--all` includes sub-agent sessions) |
| `kestrel trace [SESSION]` | Timeline, token totals, tool usage and recovery actions |
| `kestrel eval evals/basic.json [--json report.json]` | Run an evaluation suite against the configured model |
| `kestrel info` | Show configuration |

### Web interface

```bash
kestrel web            # then open the printed link: http://127.0.0.1:8765/#token=...
```

* **Chat** streams tokens, shows every tool call as a collapsible card (arguments, result, error state), and
  renders a safe subset of Markdown. When a risky tool needs approval, an **Allow / Deny** card appears in the
  conversation; the run waits for you (and denies after 2 minutes, or if no browser is attached).
  **Stop** cancels the running turn; closing the tab does too. The log makes a cancelled turn resumable.
* **Trace** shows model/tool time, tokens, estimated cost, recovery actions and the full event timeline.
  (Tool durations include time spent waiting for your approval.)
* **Memory** edits the persona, lists/adds/removes long-term facts, and runs the same hybrid search the agent uses,
  with dense and BM25 scores visible.
* **Tools & skills** lists every registered tool with its risk level, discovered skills and MCP connection status.
* **Sub-agents** lists delegated jobs and lets you cancel running ones.
* The approval mode (ask / auto-run / read-only) can be switched live from the header.

Because the agent can run shell commands and write files, the server is locked down: an access token (random per
start, or `--token` / `KESTREL_WEB_TOKEN`) is required for every API call and is sent in a header, never in a
URL; the `Host` header must match a local name (DNS-rebinding defence); cross-origin writes are refused; a strict
Content-Security-Policy forbids inline scripts. It binds to `127.0.0.1` by default. Binding elsewhere
(`--host 0.0.0.0 --allow-host my-machine`) exposes shell access to anyone holding the token, so put it behind TLS
and a trusted network first.

### Telegram

```bash
# 1. Create a bot with @BotFather and copy its token.
# 2. In .env (the allow-list is mandatory):
KESTREL_TELEGRAM_TOKEN=123456:ABC...
KESTREL_TELEGRAM_ALLOWED=your_username,123456789     # usernames (no @) and/or numeric user ids
# 3. Start the web UI; the Telegram channel starts with it
kestrel web
```

`kestrel telegram` runs the channel on its own (no web UI). `kestrel info` shows whether it is configured.

* **One agent, two front ends.** A Telegram chat is a session named `tg:<chat_id>` in the same database, so it appears
  (with a TG badge) in the web UI and can be inspected in Trace. Long-term and episodic memory are shared with every other
  session. `/new` starts a fresh conversation, `/stop` cancels the running turn, `/help` lists commands.
* **Deny by default.** Only listed users are answered, only in private chats; everyone else is ignored (and logged, so you
  can see their id). With an empty allow-list the channel refuses to start, because the agent can run commands.
* **Approvals in the chat.** In `ask` mode a risky tool sends an Allow / Deny button message to the chat that triggered it
  and waits (2 minutes, then denies). Only an allowed user in that same chat can answer.
* Replies are plain text, split to fit Telegram's size limit. Updates are acknowledged before they are handled, so a restart
  never replays old messages. The bot token is redacted from logs and error messages.
* Use a bot that nothing else is polling; Telegram allows one poller per bot and reports a conflict otherwise.
* A message typed in the web UI into a Telegram session is answered in the web UI only; it is not forwarded to Telegram.

### Using MCP servers

```bash
cp mcp_servers.example.json mcp_servers.json     # edit as needed (Windows cmd.exe: copy; PowerShell accepts cp)
kestrel chat --mcp mcp_servers.json
```

The example starts `mcp-server-time` with `uvx`, so it needs [uv](https://docs.astral.sh/uv/) installed (servers started with `npx` need Node.js).

Each server entry has `command`, `args`, optional `env`, and a `risk` of `read` (no approval needed) or
`write` (default; goes through the approval policy). Tools appear as `<server>__<tool>`.

### Adding a skill

Create `<workspace>/skills/<name>/SKILL.md` (or `~/.kestrel/skills/`). Later directories override earlier ones:
bundled → user → workspace. Skills are discovered on every message, so no restart is needed. A skill's `references/`
files can be read from any of these directories (read-only); writes stay confined to the workspace.

```markdown
---
name: my-skill
description: One line that tells the model when to use this skill.
requires:
  bins: [git]
---
Instructions the model follows once it loads the skill. Relative resources resolve from the skill folder.
```

## Architecture

```
                       +-------------------------------+
   user / CLI -------> |            Agent loop         | <---- settings, persona file
                       |  recall -> settle -> compact  |
                       |  -> model -> commit -> tools  |
                       +---+-----------+-----------+---+
                           |           |           |
                  +--------v--+   +----v-----+  +--v-----------+
                  | Tool      |   | Context  |  | Memory       |
                  | registry  |   | builder  |  | episodic     |
                  | + policy  |   | + compac-|  | long-term    |
                  +--+--+--+--+   |   tion   |  | skills       |
                     |  |  |      +----+-----+  +------+-------+
        builtin -----+  |  +---- MCP   |               |
        sub-agents -----+        servers|               |
                           +-----------v---------------v--+
                           |  SQLite: messages (append-only),
                           |  kv (versioned), events, episodes
                           +-------------------------------+
```

### Design decisions worth knowing

1. **The log is the state.** There is no separate in-memory run state. Everything the loop needs is
   recomputed from the message log, which is why a crashed run can be resumed.
2. **Commit, then execute.** An assistant message containing tool calls is persisted *before* any tool
   runs. After a crash, the unanswered calls can be derived exactly. On resume, idempotent tools are
   re-run; every other tool gets a synthetic "outcome unknown, inspect state first" result instead of being
   blindly repeated.
3. **Tool arguments are untrusted input.** Parse, validate, authorise, run with a timeout, truncate,
   label. Any failure becomes a tool *result* so the model can self-correct instead of the run dying.
4. **Compaction is a projection, not an edit.** A `summary` row records which rows it covers; the prompt
   shows the summary instead. The original conversation stays queryable, and compaction can be re-done.
5. **Retrieval is hybrid on purpose.** Embeddings blur exact identifiers (ticket numbers, names); BM25
   misses paraphrases. Fusing ranks gets both without tuning score scales.
6. **Memory failures degrade, never crash.** If the embedding service is down, the turn simply runs without
   recall.
7. **Delegation is constrained.** Children get a restricted tool profile, a smaller step budget, cannot
   spawn more agents, and cannot write to episodic memory.

## Configuration

All settings come from environment variables (or `.env`). The most useful ones:

| Variable | Default | Meaning |
|---|---|---|
| `KESTREL_BASE_URL` / `KESTREL_API_KEY` / `KESTREL_MODEL` | DeepSeek / – / `deepseek-chat` | Any OpenAI-compatible chat endpoint |
| `KESTREL_MAX_TOKENS_PARAM` | `max_tokens` | Set to `max_completion_tokens` for OpenAI reasoning-family models |
| `KESTREL_EMBED_*` | OpenAI `text-embedding-3-small` | Embedding endpoint; empty key uses an offline hashing fallback |
| `KESTREL_APPROVAL` | `ask` | `ask`, `auto` or `readonly` |
| `KESTREL_CONTEXT_WINDOW` | `64000` | Used to decide when to compact (at 70% of the window) |
| `KESTREL_HOME` | `~/.kestrel` | Database, workspace, memory files |
| `KESTREL_MCP_CONFIG` | – | Path to an MCP servers file |

State lives in `$KESTREL_HOME`: `state.db` (SQLite) and `workspace/` (the agent's sandbox, including
`memory/PERSONA.md` and `memory/MEMORY.md`, which you can edit by hand; edits apply on the next run).

## Platform notes

Developed on macOS. CI (`.github/workflows/ci.yml`) runs the test suite on **Linux, macOS and Windows** for every push, so
the current status per platform is whatever the latest Actions run shows; this README does not claim more than that.

* **`shell` tool**: runs through the platform shell, so it is `cmd.exe` on Windows and `sh` elsewhere (the tool description
  tells the model which one). Timeouts kill the whole process tree (`taskkill /T` on Windows, a process-group kill elsewhere).
* **Deny-list**: covers destructive Unix commands and the common Windows equivalents (`rd /s /q C:\`, `format C:`,
  `Remove-Item -Recurse` on a drive root, `iwr ... | iex`). It is a seat belt, not a sandbox, on every platform.
* **MCP launchers** such as `npx` are resolved through `PATH`/`PATHEXT` on Windows (they are `.cmd` files there).
* Terminal output is forced to UTF-8, so emoji and non-Latin text do not crash a legacy Windows console.

## Testing

```bash
pytest
```

145 tests run fully offline using a scripted model, an in-process fake HTTP transport, and a real stdio MCP
server. They cover validation edge cases, retry and streaming behaviour, crash recovery, concurrency,
compaction boundaries, retrieval, sub-agents, guardrails, the CLI and the web API (token, Host and Origin checks,
streaming events, approval round-trips, timeouts, cancellation). Spot-checks that deliberately break
key behaviours (skipping crash settlement, splitting tool pairs during compaction, retrying on HTTP 400,
ignoring version checks, executing tools before committing, and removing each web security check or making approvals
default to "allow") are each caught by at least one test.

`python examples/memory_trials.py 6` repeats the long-term-memory scenarios (saving without being asked, updating a fact,
recalling in a new session) against your configured model and prints counts, because a single run proves little.

Not covered: the browser front end has no automated tests. It was exercised by hand against a fake provider (streaming,
Markdown rendering with an injected `<script>`, approve and deny flows, trace, memory, tools and narrow-screen layouts).
Also not covered: behaviour against a live model provider. Prompts and tool descriptions are tuned for
tool-calling chat models but have only been exercised against scripted and fake servers; run
`kestrel eval evals/basic.json` with your own key to measure a real model.

## Limitations

- **The shell deny-list is a seat belt, not a sandbox** (on every platform). `shell` runs real commands as your user. Use
  `ask` mode (default) unless you trust the model and the prompt source.
- Token counting is a character-based estimate (`chars / 3`), deliberately conservative.
- Episodic search scans all stored vectors in memory; fine up to tens of thousands of exchanges, not millions.
- Telegram: text only, private chats only, long polling (no webhook), plain-text replies; the bot cannot be shared between two
  running programs.
- Single process, single SQLite file. Per-session runs are serialised; there is no multi-user auth.
- MCP support is stdio only.
- In the web UI each chat runs one turn at a time (a second message gets HTTP 409); approvals expire after 2 minutes;
  there is no user management beyond the access token.
- Background sub-agent results are delivered at the start of the parent's next run, not pushed mid-run.
- Reasoning-mode models that return a separate reasoning field are supported for output, but the
  reasoning is not replayed to the provider on later turns.

## Layout

```
src/kestrel/
  agent.py        the loop, tool execution, crash recovery
  app.py          composition root
  cli.py          commands
  common.py       shared types and errors
  config.py       settings
  context.py      prompt assembly, compaction
  evals.py        evaluation harness
  mcp_client.py   MCP client
  skills.py       skill discovery and loading
  store.py        SQLite (log, kv, events, episodes)
  subagents.py    delegation
  tracing.py      events and trace summaries
  llm/            chat client, embeddings, scripted model
  memory/         episodic search, BM25, long-term file
  tools/          registry, guardrails, built-in tools
  channels/       Telegram channel
  web/            HTTP API, run/approval manager, static single-page UI
skills/           bundled example skill
evals/            example eval suite
examples/         offline demo
tests/            test suite
```
