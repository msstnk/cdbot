from __future__ import annotations

import ast
import json
from pathlib import Path
from string import Formatter

from bot.localization import DEFAULT_LOCALE, DEFAULT_LOCALES_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOT_DIR = PROJECT_ROOT / "bot"


class _MessageCallCollector(ast.NodeVisitor):
    """Collect statically resolvable Messages.text/format usage."""

    def __init__(self) -> None:
        self.text_keys: set[str] = set()
        self.format_calls: dict[str, set[frozenset[str]]] = {}

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"text", "format"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            key = node.args[0].value
            if node.func.attr == "text":
                self.text_keys.add(key)
            else:
                kwargs = frozenset(
                    keyword.arg for keyword in node.keywords if keyword.arg is not None
                )
                self.format_calls.setdefault(key, set()).add(kwargs)
        self.generic_visit(node)


def test_locales_cover_used_message_keys() -> None:
    catalogs = _load_catalogs()
    used_keys = _used_locale_keys()

    for locale, messages in catalogs.items():
        missing = sorted(used_keys - messages.keys())
        assert not missing, f"Locale {locale} is missing keys: {missing}"


def test_locales_share_same_key_set() -> None:
    catalogs = _load_catalogs()
    default_keys = set(catalogs[DEFAULT_LOCALE])

    for locale, messages in catalogs.items():
        extra = sorted(messages.keys() - default_keys)
        missing = sorted(default_keys - messages.keys())
        assert not extra, f"Locale {locale} has extra keys: {extra}"
        assert not missing, f"Locale {locale} is missing keys: {missing}"


def test_default_locale_has_no_unused_keys() -> None:
    catalogs = _load_catalogs()
    used_keys = _used_locale_keys()
    default_keys = set(catalogs[DEFAULT_LOCALE])

    unused = sorted(default_keys - used_keys)
    assert not unused, f"Default locale has unused keys: {unused}"


def test_message_calls_use_literal_locale_keys() -> None:
    violations: list[str] = []
    for path in BOT_DIR.rglob("*.py"):
        if path.name in {"__init__.py", "localization.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"text", "format"} or not node.args:
                continue
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                continue
            violations.append(
                f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} uses "
                f"{node.func.attr}() with a non-literal locale key"
            )
    assert not violations, "Locale keys must be string literals:\n" + "\n".join(violations)


def test_text_calls_target_templates_without_placeholders() -> None:
    catalogs = _load_catalogs()
    collector = _collect_message_calls()

    text_only_keys = collector.text_keys - set(collector.format_calls)
    for key in sorted(text_only_keys):
        for locale, messages in catalogs.items():
            placeholders = _placeholders(messages[key])
            assert not placeholders, (
                f"Locale {locale} key {key} has placeholders {sorted(placeholders)} "
                "but is rendered with text()"
            )


def test_format_calls_match_template_placeholders() -> None:
    catalogs = _load_catalogs()
    collector = _collect_message_calls()

    for key, kwargs_sets in sorted(collector.format_calls.items()):
        assert len(kwargs_sets) == 1, (
            f"Locale key {key} is formatted with inconsistent kwargs: "
            f"{sorted(sorted(kwargs) for kwargs in kwargs_sets)}"
        )
        expected_kwargs = next(iter(kwargs_sets))
        for locale, messages in catalogs.items():
            placeholders = _placeholders(messages[key])
            assert placeholders == expected_kwargs, (
                f"Locale {locale} key {key} expects placeholders "
                f"{sorted(placeholders)} but call sites pass {sorted(expected_kwargs)}"
            )


def _collect_message_calls() -> _MessageCallCollector:
    collector = _MessageCallCollector()
    for path in BOT_DIR.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        collector.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    return collector


def _load_catalogs() -> dict[str, dict[str, str]]:
    data = json.loads(DEFAULT_LOCALES_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return {
        locale: messages
        for locale, messages in data.items()
        if isinstance(locale, str) and isinstance(messages, dict)
    }


def _placeholders(template: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name:
            fields.add(field_name)
    return fields


def _used_locale_keys() -> set[str]:
    collector = _collect_message_calls()
    return collector.text_keys | set(collector.format_calls)
