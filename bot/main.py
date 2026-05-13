"""Entrypoint for running the Codex Discord bot."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from .config import Settings
from .debug_logging import configure_debug_logger
from .discord_bot import CodexDiscordBot


def main() -> None:
    """Load configuration and start the Discord client."""
    load_dotenv(Path(".env"))
    try:
        settings = Settings.from_env()
        logger = configure_debug_logger(
            level_name=settings.debug.level_name,
            log_path=settings.storage.debug_log_path,
        )
    except RuntimeError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    logger.info(
        "cdbot.startup level=%s log_path=%s",
        settings.debug.level_name,
        settings.storage.debug_log_path,
    )
    bot = CodexDiscordBot(settings)
    bot.run(settings.discord.bot_token)
