"""Integration tests for the JutulDarcy wrapper.

These tests execute a real simulation and are intentionally heavier than unit tests.
Run selectively when validating simulator integration.
"""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import os
import shutil
import numpy as np
import pytest

from subsurface.multphaseflow.jutul_darcy import JutulDarcy


REPORT_DATES = [
    datetime(2023, 2, 5),
    datetime(2024, 3, 11),
    datetime(2025, 4, 15),
    datetime(2026, 5, 20),
    datetime(2027, 6, 24),
    datetime(2028, 7, 28),
    datetime(2029, 9, 1),
    datetime(2030, 10, 6),
    datetime(2031, 11, 10),
    datetime(2032, 12, 14),
]

DATA_TYPES = [
    "WOPR:PRO1",
    "WOPR:PRO2",
    "WOPR:PRO3",
    "WWPR:PRO1",
    "WWPR:PRO2",
    "WWPR:PRO3",
    "WWIR:INJ1",
]

GRADIENT_STEP = datetime(2032, 12, 14)
GRADIENT_TARGET = "WOPR:PRO2"


def _tiny_folder() -> Path:
    """Return absolute path to the `Example/TINY` input case directory."""
    tiny_path = Path(__file__).resolve().parents[1] / "Example" / "TINY"
    if not tiny_path.exists():
        raise FileNotFoundError(f"TINY folder not found at: {tiny_path}")
    return tiny_path


@contextmanager
def _working_directory(path: Path):
    """Temporarily change current working directory."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _copy_case_folder(target: Path, source: Path, name: str) -> Path:
    """Copy the TINY case folder into a per-test destination."""
    case_path = target / name
    shutil.copytree(source, case_path)
    return case_path


def _run_case(case_path: Path, options: dict):
    """Run JutulDarcy for the given case path and options."""
    log_permx = np.log(np.load(case_path / "PERMX.npy"))
    simulator = JutulDarcy(options)
    with _working_directory(case_path):
        return simulator({"log_permx": log_permx})


@pytest.fixture(scope="module")
def options() -> dict:
    """Base simulation options shared by tests."""
    return {
        "reporttype": "dates",
        "reportpoint": REPORT_DATES,
        "runfile": "RUNFILE.mako",
        "datatype": DATA_TYPES,
    }


@pytest.fixture(scope="module")
def tiny_folder() -> Path:
    """Path to the immutable source TINY case in repository."""
    return _tiny_folder()


@pytest.fixture(scope="module")
def run_simulation_with_adjoint(tmp_path_factory, options, tiny_folder):
    """Run one adjoint-enabled simulation and share output across gradient tests."""
    base_tmp = tmp_path_factory.mktemp("adjoint")
    case_path = _copy_case_folder(base_tmp, tiny_folder, "TINY_ADJOINT")

    case_options = {
        **options,
        'perm_copied': True, # Include total derivative when PERMX is copied to PERMY and PERMZ
        "adjoints": {
            "WOPR": {
                "steps": [GRADIENT_STEP],
                "wellID": "PRO2",
                "parameters": ["log_permx", "permx"],
            }
        },
    }
    return _run_case(case_path, case_options)


def test_simulation_runs_and_matches_requested_outputs(tmp_path, options, tiny_folder):
    """Simulation returns a non-empty table with all requested columns and dates."""
    case_path = _copy_case_folder(tmp_path, tiny_folder, "TINY")
    results = _run_case(case_path, options)

    assert not results.empty, "Simulation result table is empty"

    missing_columns = sorted(set(options["datatype"]) - set(results.columns))
    assert not missing_columns, f"Missing expected result columns: {missing_columns}"

    missing_dates = [date for date in options["reportpoint"] if date not in results.index]
    assert not missing_dates, f"Missing expected report dates: {missing_dates}"


def test_gradient_contains_expected_structure(run_simulation_with_adjoint, options):
    """Adjoint run returns expected gradient columns and valid report dates."""
    _, gradient = run_simulation_with_adjoint

    assert not gradient.empty, "Gradient table is empty"

    expected_columns = {
        (GRADIENT_TARGET, "log_permx"),
        (GRADIENT_TARGET, "permx"),
    }
    missing_columns = sorted(expected_columns - set(gradient.columns))
    assert not missing_columns, f"Missing expected gradient columns: {missing_columns}"

    unexpected_dates = [date for date in gradient.index if date not in options["reportpoint"]]
    assert not unexpected_dates, f"Gradient has unexpected report dates: {unexpected_dates}"


def test_gradient_log_permx_matches_chain_rule(run_simulation_with_adjoint, tiny_folder):
    """Validate dF/d(log(k)) = dF/d(k) * k for the configured objective and step."""
    _, gradient = run_simulation_with_adjoint

    grad_log_permx = gradient.loc[GRADIENT_STEP, (GRADIENT_TARGET, "log_permx")]
    grad_permx = gradient.loc[GRADIENT_STEP, (GRADIENT_TARGET, "permx")]
    permx = np.load(tiny_folder / "PERMX.npy")

    np.testing.assert_allclose(grad_log_permx, grad_permx * permx)


