'''Tests for JutulDarcy wrapper. NB! These tests are time-consuming and should be run selectively.'''
import pytest
import shutil
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from subsurface.multphaseflow.jutul_darcy import JutulDarcy

def _tiny_folder() -> Path:
    # /.../SimulatorWrap/tests/test_jutul_darcy.py -> /.../SimulatorWrap/Example/TINY
    p = Path(__file__).resolve().parents[1] / "Example" / "TINY"
    if not p.exists():
        raise FileNotFoundError(f"TINY folder not found at: {p}")
    return p

@pytest.fixture(scope="module")
def options():
    datetimes = [
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

    datatype = [
        'WOPR:PRO1', 'WOPR:PRO2', 'WOPR:PRO3',
        'WWPR:PRO1', 'WWPR:PRO2', 'WWPR:PRO3',
        'WWIR:INJ1'
    ]

    return {
        'reporttype': 'dates',
        'reportpoint': datetimes,
        'runfile': 'RUNFILE.mako',
        'startdate': datetime(2022, 1, 1),
        'datatype': datatype,
    }

@pytest.fixture(scope="module")
def run_simulation_with_adjoint(tmp_path_factory, options):
    """Compute adjoint gradient once for all tests in this module."""
    tmp_path = tmp_path_factory.mktemp("adjoint")

    tiny_folder = _tiny_folder()
    folder = tmp_path / 'TINY_ADJOINT'
    shutil.copytree(tiny_folder, folder)

    # Do not mutate shared options fixture
    opts = dict(options)
    opts['adjoints'] = {
        'WOPR': {
            'steps': [datetime(2032, 12, 14)],
            'wellID': 'PRO2',
            'parameters': ['log_permx', 'permx']
        }
    }

    cwd = os.getcwd()
    try:
        os.chdir(folder)
        log_permx = np.log(np.load(folder / 'PERMX.npy'))
        simulator = JutulDarcy(opts)
        results, gradient = simulator({'log_permx': log_permx})
    finally:
        os.chdir(cwd)

    return results, gradient
    

class TestJutulDarcySimulation:

    def _copy_TINY_folder(self, tmp_path):
        """Copy the TINY folder to a temporary directory for testing."""
        tiny_folder = _tiny_folder()
        dest_folder = tmp_path / 'TINY'
        shutil.copytree(tiny_folder, dest_folder)
        return dest_folder

    def test_simulation_runs(self, tmp_path, options):
        '''Test that the JutulDarcy simulation runs and produces results.'''
        # Copy TINY folder to temporary directory
        folder = self._copy_TINY_folder(tmp_path)
        os.chdir(folder)

        # Load log_permx and run simulation
        log_permx = np.log(np.load(folder / 'PERMX.npy'))
        simulator = JutulDarcy(options)
        result = simulator({'log_permx': log_permx})[0]

        # Check that results are not empty
        assert not result.empty, "Simulation results are empty"

        # Check that results contain expected columns
        expected_columns = options['datatype']
        for col in expected_columns:
            assert col in result.columns, f"Missing expected column: {col}"
        
        # Check that results contain expected time steps
        expected_index = options['reportpoint']
        for time in expected_index:
            assert time in result.index, f"Missing expected time step: {time}"


class TestJutulDarcyGradient:

    def test_gradient_computation(self, run_simulation_with_adjoint, options):
        """Test that the gradient is computed and has the expected structure."""
        results, gradient = run_simulation_with_adjoint
        
        # Check that gradient is not empty
        assert not gradient.empty, "Gradient results are empty"

        # Check that gradient contains expected columns
        expected_columns = [('WOPR:PRO2', 'log_permx')]
        for col in expected_columns:
            assert col in gradient.columns, f"Missing expected gradient column: {col}"
        
        # Check that gradient contains expected time steps
        for index in gradient.index:
            assert index in options['reportpoint'], f"Missing expected time step in gradient: {index}"
    
    def test_gradient_consistency(self, run_simulation_with_adjoint):
        """Test that the gradient with respect to log_permx is consistent with the gradient with respect to permx."""
        results, gradient = run_simulation_with_adjoint
        
        # Extract gradients for log_permx and permx
        grad_log_permx = gradient.loc[datetime(2032, 12, 14), ('WOPR:PRO2', 'log_permx')]
        grad_permx = gradient.loc[datetime(2032, 12, 14), ('WOPR:PRO2', 'permx')]
        
        # Load original permx values
        permx = np.load(_tiny_folder() / 'PERMX.npy')
        
        # Check consistency: grad_log_permx should equal grad_permx * permx
        np.testing.assert_almost_equal(grad_log_permx, grad_permx * permx)
    
        
        