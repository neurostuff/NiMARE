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


def test_float64_finfo_survives_a_later_nibabel_import():
    """``np.finfo(float)`` must describe float64 even when nibabel is imported after NiMARE.

    NumPy caches ``finfo`` objects by dtype, and where long double is the same size as double
    (Windows) the two dtypes compare equal, so whichever is requested first answers for both.
    nibabel requests long double on import; if that comes first, ``np.finfo(float).eps`` is a
    long double scalar, and libraries that read it at import (statsmodels' GLM links) promote
    their arrays to long double, which ``numpy.linalg`` rejects.
    """
    code = "import nimare, nibabel, numpy as np; print(type(np.finfo(float).eps).__name__)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "float64"


def test_lazy_attributes_resolve():
    """Attribute access through the lazy packages returns the defining module's objects."""
    import nimare
    from nimare.meta.cbma import ALE

    assert nimare.meta.ALE is ALE
    assert nimare.meta.cbma.ALE is ALE
    assert nimare.correct.FWECorrector.__module__ == "nimare.correct"
    with pytest.raises(AttributeError):
        nimare.not_a_submodule
