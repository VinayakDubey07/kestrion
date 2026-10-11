"""
Unit tests for kestrion.core.compat deprecation utilities.
"""

import pytest

from kestrion.core.compat import deprecated


def test_deprecated_function_emits_warning():
    @deprecated(since="0.8.0", use="new_func()")
    def old_func(a: int, b: int = 10) -> int:
        """Old function docstring."""
        return a + b

    with pytest.deprecated_call() as warning_info:
        result = old_func(5, b=15)

    assert result == 20
    assert len(warning_info) == 1
    assert "old_func is deprecated since Kestrion v0.8.0" in str(warning_info[0].message)
    assert "new_func()" in str(warning_info[0].message)
    assert ".. deprecated::" in old_func.__doc__
    assert "Old function docstring." in old_func.__doc__


def test_deprecated_class_emits_warning_on_instantiation():
    @deprecated(since="0.8.0", use="NewService")
    class OldService:
        """Old service docstring."""

        def __init__(self, name: str):
            self.name = name

        def greet(self) -> str:
            return f"Hello, {self.name}"

    with pytest.deprecated_call() as warning_info:
        instance = OldService("enterprise")

    assert instance.name == "enterprise"
    assert instance.greet() == "Hello, enterprise"
    assert len(warning_info) == 1
    assert "OldService is deprecated since Kestrion v0.8.0" in str(warning_info[0].message)
    assert "NewService" in str(warning_info[0].message)
    assert ".. deprecated::" in OldService.__doc__


def test_deprecated_with_custom_category():
    @deprecated(since="1.0.0", use="Agent.run_async()", category=FutureWarning)
    def legacy_runner() -> str:
        return "ok"

    with pytest.warns(FutureWarning, match="legacy_runner is deprecated"):
        res = legacy_runner()

    assert res == "ok"
