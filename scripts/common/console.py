"""Compatibility re-exports for shared console utilities."""

from glucose_forecasting.common.console import (
    configure_stdio_utf8,
    init_cli_console,
    safe_echo,
)

__all__ = ["configure_stdio_utf8", "init_cli_console", "safe_echo"]
