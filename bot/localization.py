"""Load localized message templates for Discord responses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOCALE = "en_US"
DEFAULT_LOCALES_PATH = Path(__file__).with_name("locales.json")


@dataclass(frozen=True, slots=True)
class Messages:
    """Resolved message templates for one locale with fallback values merged in."""

    locale: str
    values: dict[str, str]

    @classmethod
    def load(
        cls,
        locale: str,
        path: str | Path = DEFAULT_LOCALES_PATH,
        *,
        fallback_locale: str = DEFAULT_LOCALE,
    ) -> Messages:
        """Load a locale catalog and merge missing keys from the fallback locale."""
        locales_path = Path(path)
        data = json.loads(locales_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError(f"Locales file must be a JSON object: {locales_path}")

        selected_locale = locale.strip() or fallback_locale
        fallback_values = _locale_values(data, fallback_locale, locales_path)
        selected_values = _locale_values(data, selected_locale, locales_path)

        merged = dict(fallback_values)
        merged.update(selected_values)
        return cls(locale=selected_locale, values=merged)

    def text(self, key: str) -> str:
        """Return a localized message template by key."""
        try:
            return self.values[key]
        except KeyError as exc:
            raise KeyError(f"Locale key was not found: {key}") from exc

    def format(self, key: str, **values: object) -> str:
        """Return a localized message with named format values applied."""
        return self.text(key).format(**values)


def load_default_messages(locale: str = DEFAULT_LOCALE) -> Messages:
    """Load messages from the package default locale file."""
    return Messages.load(locale, DEFAULT_LOCALES_PATH)


def _locale_values(
    data: dict[str, object],
    locale: str,
    locales_path: Path,
) -> dict[str, str]:
    raw_locale = data.get(locale)
    if not isinstance(raw_locale, dict):
        available = ", ".join(sorted(str(key) for key in data))
        raise RuntimeError(
            f"Locale `{locale}` was not found in {locales_path}. Available locales: {available}"
        )

    values: dict[str, str] = {}
    for key, value in raw_locale.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise RuntimeError(
                f"Locale `{locale}` in {locales_path} must contain only string keys and values"
            )
        values[key] = value
    return values
