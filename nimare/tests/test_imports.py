"""Test that importing NiMARE stays lazy."""

import subprocess
import sys

import pytest


def _modules_after(statement):
    """Return the modules a fresh interpreter has loaded after running ``statement``."""
    code = f"import sys; {statement}; print('\\n'.join(sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    return set(out.split())


@pytest.mark.parametrize(
    "statement,absent",
    [
        ("import nimare", ["nimare.meta", "nilearn", "scipy", "matplotlib", "pymare", "torch"]),
        (
            "from nimare.meta.cbma import ALE",
            ["nimare.meta.ibma", "pymare", "sympy", "matplotlib", "plotly", "torch"],
        ),
    ],
)
def test_import_does_not_load_unneeded_modules(statement, absent):
    """Subpackages, and the heavy dependencies only some of them need, load on first use."""
    loaded = _modules_after(statement)
    assert not loaded & set(absent)


def test_lazy_attributes_resolve():
    """Attribute access through the lazy packages returns the defining module's objects."""
    import nimare
    from nimare.meta.cbma import ALE

    assert nimare.meta.ALE is ALE
    assert nimare.meta.cbma.ALE is ALE
    assert nimare.correct.FWECorrector.__module__ == "nimare.correct"
    with pytest.raises(AttributeError):
        nimare.not_a_submodule
