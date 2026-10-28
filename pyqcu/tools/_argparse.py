"""Small argparse compatibility helpers for supported Python 3.8 runtimes."""

from __future__ import annotations

import argparse


class BooleanOptionalAction(argparse.Action):
    """Backport of :class:`argparse.BooleanOptionalAction` for Python 3.8."""

    def __init__(
        self,
        option_strings,
        dest,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None,
    ):
        expanded = []
        for option_string in option_strings:
            expanded.append(option_string)
            if option_string.startswith("--"):
                expanded.append("--no-" + option_string[2:])
        if help is not None and default is not None:
            help += f" (default: {default})"
        super().__init__(
            option_strings=expanded,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string is not None:
            value = not option_string.startswith("--no-")
            setattr(namespace, self.dest, value)

    def format_usage(self):
        return " | ".join(self.option_strings)


def install_boolean_optional_action():
    """Install the fallback only when the standard library lacks it."""
    if not hasattr(argparse, "BooleanOptionalAction"):
        argparse.BooleanOptionalAction = BooleanOptionalAction
    return argparse.BooleanOptionalAction


install_boolean_optional_action()


__all__ = ["BooleanOptionalAction", "install_boolean_optional_action"]
