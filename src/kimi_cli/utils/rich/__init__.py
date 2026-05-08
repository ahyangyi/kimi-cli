"""Project-wide Rich configuration helpers."""

from __future__ import annotations

import re
from typing import Final

from rich import _wrap
from rich.box import Box

# Custom box style used for all Panel instances in the CLI.
INTEGRAL_BOX = Box("╭┄┄⌠\n│ ││\n╞═╪╡\n│─┼│\n│─┼│\n╞═╪╡\n│ ││\n⌡┄┄╯\n")

# Regex used by Rich to compute break opportunities during wrapping.
_DEFAULT_WRAP_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s*\S+\s*")
_CHAR_WRAP_PATTERN: Final[re.Pattern[str]] = re.compile(r".", re.DOTALL)


def enable_character_wrap() -> None:
    """Switch Rich's wrapping logic to break on every character.

    Rich's default behavior tries to preserve whole words; we override the
    internal regex so markdown rendering can fold text at any column once it
    exceeds the terminal width.
    """

    _wrap.re_word = _CHAR_WRAP_PATTERN


def restore_word_wrap() -> None:
    """Restore Rich's default word-based wrapping."""

    _wrap.re_word = _DEFAULT_WRAP_PATTERN


# Apply character-based wrapping globally for the CLI.
enable_character_wrap()

# Monkey-patch Panel to use INTEGRAL_BOX by default instead of ROUNDED.
from rich.panel import Panel as _Panel  # noqa: E402

_original_panel_init = _Panel.__init__


def _patched_panel_init(self, *args, box=INTEGRAL_BOX, **kwargs):  # type: ignore[no-untyped-def]
    _original_panel_init(self, *args, box=box, **kwargs)  # pyright: ignore[reportUnknownArgumentType]


_Panel.__init__ = _patched_panel_init  # type: ignore[method-assign]
