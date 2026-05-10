# cdbot

![Python](https://img.shields.io/badge/python-3.13+-3776AB?logo=python&logoColor=white)
![Discord](https://img.shields.io/badge/discord-DM%20bot-5865F2?logo=discord&logoColor=white)
![Codex](https://img.shields.io/badge/codex-app--server-black)
![Status](https://img.shields.io/badge/status-MVP-2ea44f)

Run Codex from Discord DMs.


`cdbot` is a lightweight bridge between Discord and the Codex app-server SDK. Send a message to the bot in a DM, and it forwards that prompt into a local Codex session, streams the response back into Discord, and keeps the thread state around for follow-up turns.

It is intentionally small, local-first, and practical: one bot process, one Codex binary, one workspace, and a clean approval flow for actions that need confirmation.

![Screenshot](assets/cdbot.png)

## Why this exists

If you already use Codex locally, Discord can become a surprisingly convenient control surface:

- Ask Codex to inspect or edit a repo from your phone or another machine.
- Keep a DM thread tied to a resumable Codex session.
- Review approval requests directly inside Discord.
- Change the active model or working directory without access the terminal.

## Features

- *DM-first interaction model*: Messages from Discord DMs are treated as Codex prompts.
- *Streaming replies*: Assistant output is forwarded back to Discord as it arrives.
- *Session Persistence & Control*: Each DM thread maps to a unique Codex session ID to maintain continuous context. Use the /clear command to terminate the current session and start a fresh one.
- *In-chat approvals*: File changes and command execution approvals are surfaced as Discord UI buttons. Additionally, you can approve for the rest of the current turn.
- *Remote Agent Management*: Issue `/model`, `/cwd` and other control commands to manage the remote Codex session without terminal access.

![In Chat Approval](assets/in_discord_app.png)

## Quick Start

### Requirements

- Python `3.13+`
- `uv`
- A working local Codex installation
- A Discord bot token with permission to receive DMs

### Setup

```bash
git clone https://github.com/msstnk/cdbot.git
cd cdbot

git clone https://github.com/openai/codex.git vendor/codex
uv sync --dev
```


### Prepare a Discord Server and Bot Token

1. **Create a private Discord server where you are the only member**. By default, the bot does not check for user permissions and will respond to any direct message (DM) from any user who shares a server with it.
2. Go to the [Discord Developer Portal](https://discord.com/developers/applications) and create a new application.
3. Under the "Bot" tab, name your bot and reset the token to get your `CDBOT_DISCORD_BOT_TOKEN`.
4. Under the "OAuth2" tab, check the "bot" scope. Leave all "Bot Permissions" unselected. Copy the generated URL and paste it in your browser.
    - *Note:* The bot only responds to DMs. You do not need to select any permissions.

### Bot Configuration

> [!IMPORTANT]
> The bot responds to any user in the same server by default, you can restrict this with the `CDBOT_WHITELISTED_USERS` environment variable.
> Discord user IDs will be stored in the session_store.jsonl file when session accepted or INFO+ debug logs when session rejected.

The app loads environment variables from `.env` in the project root. Create this file with the following content:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `CDBOT_DISCORD_BOT_TOKEN` | Yes | - | Discord bot token. |
| `CDBOT_CODEX_HOME` | No | `.codex` | Codex home directory passed to the runtime. |
| `CDBOT_CODEX_BIN` | No | `.codex/bin/codex` | Path to the local Codex binary. Must exist. |
| `CDBOT_CODEX_MODEL` | No | `gpt-5.5` | Default model used for new turns. |
| `CDBOT_APPROVAL_TIMEOUT_SEC` | No | `60` | Timeout for approval requests before they default to deny. |
| `CDBOT_SESSION_STORE_PATH` | No | `.local/session_store.jsonl` | JSONL file used to persist DM session state, including the Discord user id for the DM. |
| `CDBOT_WORKSPACE_CWD` | No | current process directory | Root workspace directory for Codex turns. |
| `CDBOT_WHITELISTED_USERS` | No | empty | Comma-separated Discord user ids allowed to use the bot. Empty means allow all users who can DM the bot. |
| `CDBOT_LOCALE` | No | `en_US` | Bot message locale. Bundled: `en_US`, `ja_JP`. |
| `CDBOT_DEBUG_LEVEL` | No | `OFF` | Debug log level: `OFF`, `ERROR`, `WARNING`, `INFO`, `DEBUG`, `TRACE`. |
| `CDBOT_DEBUG_LOG_PATH` | No | `.local/cdbot.log` | Debug log output path. |

### Run the bot
Run the bot simply with:

```bash
uv run main.py
```

or create a systemd service file like:

```ini cdbot.service
[Unit]
Description=cdbot service
After=network.target

[Service]
Type=simple
User=user
WorkingDirectory=/home/user/cdbot
ExecStart=/home/user/cdbot/.venv/bin/python3 /home/user/cdbot/main.py
Restart=never

[Install]
WantedBy=multi-user.target
```


## DM Commands

These are regular DM messages, not Discord slash-command registrations.

| Command | Description |
| --- | --- |
| `/clear` | Forget the saved Codex thread for the current DM while keeping per-DM settings like `cwd` and `model`. |
| `/cwd` | Show the current working directory for this DM. |
| `/cwd <path>` | Change the working directory for future turns. The path must stay inside the configured workspace root. |
| `/model` | Show the currently selected model. |
| `/model <name>` | Change the model and clear the session. |

## Notes

- The bot only responds to direct messages, not guild channels.
- Working directory changes are intentionally constrained to the configured workspace root.
- `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0` may be required on Ubuntu 24.04 to allow Codex to run bubblewrapped commands without errors, see [Codex issue#17337](https://github.com/openai/codex/issues/17337#issuecomment-4322840642) and [Codex issue#14919](https://github.com/openai/codex/issues/14919)

## License

This project is licensed under the Apache License, Version 2.0.
