"""
Compatibility utilities for managing API deprecations across Kestrion versions.

Usage::

    from kestrion.core.compat import deprecated

    @deprecated(since="0.7.0", use="Agent(output_schema=...)")
    def old_json_mode_fn():
        ...

    # Classes work too
    @deprecated(since="0.7.0", use="SQLiteCheckpointStore")
    class OldStore:
        ...

Emits a standard ``DeprecationWarning`` (hidden by default in production,
visible in test suites that enable warnings). Consumers can silence them
with ``warnings.filterwarnings("ignore", category=DeprecationWarning,
module="kestrion")``.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(
    since: str,
    use: str,
    *,
    stacklevel: int = 2,
    category: type[Warning] = DeprecationWarning,
) -> Callable[[F], F]:
    """
    Decorator to mark a callable or class as deprecated.

    Args:
        since:      The version string when the deprecation was introduced (e.g. "0.7.0").
        use:        Human-readable description of the replacement API.
        stacklevel: Passed to ``warnings.warn``; controls which frame appears
                    in the warning message. 2 (default) points at the caller of
                    the deprecated function, which is usually what you want.
        category:   Warning category. Defaults to ``DeprecationWarning``.
                    Use ``FutureWarning`` for features that will change in the
                    *next* major version.

    Returns:
        A decorator that wraps the target and emits the warning on first call.

    Example::

        @deprecated(since="0.7.0", use="NewClass")
        class OldClass:
            pass

        @deprecated(since="0.8.0", use="agent.run()", category=FutureWarning)
        def legacy_run(prompt):
            ...
    """

    def decorator(obj: F) -> F:
        msg = (
            f"{obj.__qualname__} is deprecated since Kestrion v{since} "
            f"and will be removed in a future version. Use {use!r} instead."
        )

        if isinstance(obj, type):
            # Class deprecation: patch __init__ so the warning fires on instantiation
            original_init = getattr(obj, "__init__")

            @functools.wraps(original_init)
            def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(msg, category=category, stacklevel=stacklevel)
                original_init(self, *args, **kwargs)

            setattr(obj, "__init__", new_init)
            setattr(obj, "__doc__", _prepend_deprecation_note(getattr(obj, "__doc__", None), msg))
            return obj  # type: ignore[return-value]

        # Function / method deprecation
        @functools.wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, category=category, stacklevel=stacklevel)
            return obj(*args, **kwargs)

        wrapper.__doc__ = _prepend_deprecation_note(obj.__doc__, msg)  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def _prepend_deprecation_note(docstring: str | None, msg: str) -> str:
    """Prepend a deprecation notice to an existing docstring."""
    note = f".. deprecated::\n    {msg}\n\n"
    if docstring:
        return note + docstring
    return note
