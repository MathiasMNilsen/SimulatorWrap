"""
Simulator wrapper for the JutulDarcy simulator.

This module provides a Python interface (:class:`JutulDarcy`) for running
JutulDarcy reservoir simulations from a single configuration dictionary.
It supports:

* Single or ensemble forward simulations (parallel via :mod:`p_tqdm`).
* Input through either a static ``.DATA`` file or a Mako-templated ``.mako``
  file rendered per ensemble member.
* Flexible result extraction (field and per-well summary keywords) with
  ``list`` / ``dict`` / ``DataFrame`` output formats.
* Adjoint sensitivity computation for well-based objectives with respect to
  reservoir parameters (porosity, permeability, optionally log-scaled or
  copied across PERMX/Y/Z).

The wrapper communicates with Julia via :mod:`juliacall`. Heavy Julia state
is created lazily inside worker processes so the class can be pickled for
multiprocessing.
"""

import os
import shutil
import warnings
import datetime as dt
import numpy as np
import pandas as pd

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from mako.template import Template
from p_tqdm import p_map
from tqdm import tqdm

__author__ = ["Mathias Methlie Nilsen"] # With help from "Claude Opus 4.7"
__all__ = ["JutulDarcy"]


# ============================================================================ #
# Environment configuration
# ============================================================================ #
# These environment variables must be set BEFORE juliacall is imported. We use
# `setdefault` so users can still override them externally.
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")
os.environ.setdefault("PYTHON_JULIACALL_OPTLEVEL", "3")

# Silence the noisy "juliacall module already imported" warning that appears
# in worker processes when Julia is re-imported.
warnings.filterwarnings("ignore", message=".*juliacall module already imported.*")


# ============================================================================ #
# Module-level constants
# ============================================================================ #
SECONDS_PER_DAY: int = 86_400

#: Shared progress-bar styling used by both forward and adjoint loops.
PBAR_OPTS: dict[str, Any] = {
    "ncols": 110,
    "colour": "#285475",
    "bar_format": (
        "{desc}: {percentage:3.0f}% [{bar}] {n_fmt}/{total_fmt} "
        "│ ⏱ {elapsed}<{remaining} │ {rate_fmt}"
    ),
    "ascii": "-◼",
}

#: Summary keywords whose values are cumulative (totals), as opposed to rates.
CUMULATIVE_KEYS: frozenset[str] = frozenset({
    "FOPT", "FGPT", "FWPT", "FWLT", "FWIT",
    "WOPT", "WGPT", "WWPT", "WLPT",
})

VALID_OUTPUT_FORMATS: frozenset[str] = frozenset({"list", "dict", "dataframe"})
VALID_ADJOINT_MODES: frozenset[str]  = frozenset({"sensitivities", "optimization"})
VALID_REPORT_TYPES: frozenset[str]   = frozenset({"days", "dates"})

#: Map summary keyword -> (phase string, is_rate). Cumulative totals
#: (``WxPT``) are marked ``is_rate=False``; instantaneous rates (``WxPR``,
#: ``WxIR``) are marked ``is_rate=True``.
PHASE_MAP: dict[str, tuple[str, bool]] = {
    "WOPT": ("oil", False),    "WGPT": ("gas", False),
    "WWPT": ("water", False),  "WLPT": ("liquid", False),
    "WOPR": ("oil", True),     "WGPR": ("gas", True),
    "WWPR": ("water", True),   "WWIR": ("water", True),
    "WLPR": ("liquid", True),
}

#: Map a phase name to the corresponding JutulDarcy rate-target type name.
RATE_ID_MAP: dict[str, str] = {
    "mass":   "TotalSurfaceMassRate",
    "liquid": "SurfaceLiquidRateTarget",
    "water":  "SurfaceWaterRateTarget",
    "oil":    "SurfaceOilRateTarget",
    "gas":    "SurfaceGasRateTarget",
    "rate":   "TotalRateTarget",
}

#: Display units for known summary keywords (metric system).
UNIT_MAP: dict[str, str] = {
    "PORO": "", "PERMX": "mD", "PERMY": "mD", "PERMZ": "mD",
    "FOPT": "Sm3", "FGPT": "Sm3", "FWPT": "Sm3", "FWLT": "Sm3", "FWIT": "Sm3",
    "FOPR": "Sm3/day", "FGPR": "Sm3/day", "FWPR": "Sm3/day",
    "FLPR": "Sm3/day", "FWIR": "Sm3/day",
    "WOPR": "Sm3/day", "WGPR": "Sm3/day", "WWPR": "Sm3/day",
    "WLPR": "Sm3/day", "WWIR": "Sm3/day",
}

#: Permeability keywords in canonical (x, y, z) order.
PERM_KEYS: tuple[str, ...] = ("PERMX", "PERMY", "PERMZ")

#: Map lower-case permeability parameter substring -> axis index in PERM_KEYS.
PERM_INDEX: dict[str, int] = {"permx": 0, "permy": 1, "permz": 2}


# ============================================================================ #
# Configuration dataclasses
# ============================================================================ #
@dataclass
class AdjointObjective:
    """
    One adjoint objective specification (one well + phase).

    Attributes
    ----------
    wellID : str
        Name of the well associated with this objective.
    phase : str
        Phase string used to pick the rate target (see :data:`RATE_ID_MAP`).
    is_rate : bool
        True for instantaneous rate objectives, False for cumulative totals.
    parameters : list[str]
        Parameters w.r.t. which the gradient should be evaluated, e.g.
        ``["PORO", "PERMX"]`` (case-insensitive; ``"log"`` enables log-scaling).
    steps : Any
        Either the string ``"all"`` (use the wrapper's global report points)
        or an explicit list of ``int`` days or :class:`datetime.datetime`
        objects at which to evaluate the objective.
    """
    wellID: str
    phase: str
    is_rate: bool
    parameters: list[str]
    steps: Any  # 'all' | list[int] | list[datetime]


# ============================================================================ #
# Helper functions (pure / stateless)
# ============================================================================ #
def get_metric_unit(key: str) -> str:
    """
    Return the metric-system unit string for a summary keyword.

    Parameters
    ----------
    key : str
        Summary keyword (case-insensitive), e.g. ``"FOPT"``.

    Returns
    -------
    str
        Unit string from :data:`UNIT_MAP`, or ``"Unknown"`` if not found.
    """
    return UNIT_MAP.get(key.upper(), "Unknown")


def _process_datatype_info(datatypes: Iterable[str]) -> list[str]:
    """
    Expand grouped datatype specs into one entry per well.

    A spec like ``"WOPR:W1:W2"`` is expanded to ``["WOPR:W1", "WOPR:W2"]``.
    Specs without a colon (field-level) are passed through unchanged.

    Parameters
    ----------
    datatypes : iterable of str
        User-supplied datatype identifiers.

    Returns
    -------
    list of str
        Normalised, one-well-per-entry list.
    """
    out: list[str] = []
    for d in datatypes:
        if ":" in d:
            base, *wells = d.split(":")
            out.extend(f"{base}:{w}" for w in wells)
        else:
            out.append(d)
    return out


def _process_adjoint_info(adjoint_info: dict) -> dict[str, AdjointObjective]:
    """
    Normalise the ``adjoints`` config dict into :class:`AdjointObjective` items.

    Each input entry may specify a single well or a list of wells; the output
    contains one :class:`AdjointObjective` per (datatype, well) pair, keyed by
    ``"<datatype>:<well>"``.

    Parameters
    ----------
    adjoint_info : dict
        Raw mapping ``{datatype: {"wellID": ..., "parameters": ..., "steps": ...}}``.

    Returns
    -------
    dict[str, AdjointObjective]
        Flattened objective specifications.
    """
    info: dict[str, AdjointObjective] = {}
    for dataID, spec in adjoint_info.items():
        # Normalise scalar values to lists for uniform iteration.
        wells = spec["wellID"]
        if isinstance(wells, str):
            wells = [wells]
        params = spec["parameters"]
        if isinstance(params, str):
            params = [params]

        phase, is_rate = PHASE_MAP[dataID]
        for w in wells:
            info[f"{dataID}:{w}"] = AdjointObjective(
                wellID=w, phase=phase, is_rate=is_rate,
                parameters=list(params), steps=spec["steps"],
            )
    return info


def _active_to_full_grid(vec: np.ndarray, actnum_vec: np.ndarray,
                         fill_value: float = 0.0) -> np.ndarray:
    """
    Embed an active-cell vector into the full grid layout.

    Parameters
    ----------
    vec : np.ndarray
        Either a vector defined only on active cells (length ``actnum_vec.sum()``)
        or already on the full grid (length ``len(actnum_vec)``).
    actnum_vec : np.ndarray
        Flat ACTNUM array (1 = active, 0 = inactive), Fortran-ordered.
    fill_value : float, optional
        Value placed at inactive cells. Defaults to ``0.0``.

    Returns
    -------
    np.ndarray
        Full-grid vector of length ``len(actnum_vec)`` and dtype ``float64``.

    Raises
    ------
    ValueError
        If ``vec`` matches neither the active nor the full grid size.
    """
    n_active = int(actnum_vec.sum())
    if len(vec) == n_active:
        full = np.full(actnum_vec.shape, fill_value, dtype=np.float64, order="F")
        full[actnum_vec == 1] = vec
        return full
    if len(vec) == len(actnum_vec):
        return np.asarray(vec, dtype=np.float64)
    raise ValueError("Parameter length does not match number of active cells")


@contextmanager
def _chdir(path: str | os.PathLike):
    """
    Context manager that temporarily changes the working directory.

    The previous working directory is always restored, including when the
    wrapped block raises an exception.

    Parameters
    ----------
    path : str or path-like
        Directory to switch into.

    Yields
    ------
    None
    """
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _suppress_julia(julia, code: str):
    """
    Evaluate a Julia expression with ``stdout`` and ``stderr`` redirected to devnull.

    Used to silence verbose JutulDarcy setup/solver messages.

    Parameters
    ----------
    julia : juliacall.Main
        The Julia main module reference.
    code : str
        A single Julia expression (will be wrapped, not multi-line statements).

    Returns
    -------
    Any
        Result of the inner Julia expression.
    """
    return julia.seval(f"""
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                {code}
            end
        end
    """)


def _get_mapping_value(obj, key_name: str, julia):
    """
    Look up ``key_name`` in a Julia mapping using both Symbol and string keys.

    JutulDarcy mappings sometimes use ``Symbol`` keys and sometimes plain
    strings; this helper tries both so callers don't have to.

    Parameters
    ----------
    obj : Any
        A Julia mapping-like object supporting ``haskey`` / ``getindex``.
    key_name : str
        Candidate key as a Python string.
    julia : juliacall.Main
        Julia main module (used to construct ``Symbol``).

    Returns
    -------
    Any or None
        The value if found; ``None`` otherwise.
    """
    for candidate in (julia.Symbol(key_name), key_name):
        try:
            if julia.haskey(obj, candidate):
                return obj[candidate]
        except Exception:
            # `haskey` may not be defined for every object type encountered.
            continue
    return None


def _extract_key_value(root_object, keys, julia):
    """
    Breadth-first search for the first matching key in nested Julia mappings.

    Useful because JutulDarcy gradient structures can nest gradients several
    layers deep (e.g. ``grad[:model][:reservoir][:porosity]``).

    Parameters
    ----------
    root_object : Any or list
        Starting Julia object(s) to search.
    keys : str or list of str
        Candidate key name(s). The first key found anywhere in the tree wins.
    julia : juliacall.Main
        Julia main module.

    Returns
    -------
    Any or None
        Value associated with the first matching key, or ``None``.
    """
    if not isinstance(keys, list):
        keys = [keys]
    if not isinstance(root_object, list):
        root_object = [root_object]

    queue = deque(root_object)
    visited: set[int] = set()  # avoid revisiting the same Julia object

    while queue:
        cur = queue.popleft()
        oid = id(cur)
        if oid in visited:
            continue
        visited.add(oid)

        # Try to match at the current level first.
        for k in keys:
            val = _get_mapping_value(cur, k, julia)
            if val is not None:
                return val

        # Otherwise, enqueue children for further exploration.
        try:
            for ck in list(julia.keys(cur)):
                try:
                    queue.append(cur[ck])
                except Exception:
                    continue
        except Exception:
            # `cur` is a leaf (no `keys` method); skip it.
            continue

    return None


def _extract_adjoint(jlgrad, jlcase, parameter: str, actnum: np.ndarray,
                     perm_copied: bool, julia) -> np.ndarray:
    """
    Extract and post-process an adjoint gradient for a single parameter.

    Handles unit scaling (mD → SI), optional log-scaling, embedding back onto
    the full grid, and the "copied permeability" convention where PERMY/PERMZ
    are duplicates of PERMX in the input deck.

    Parameters
    ----------
    jlgrad : Any
        Julia gradient object returned by the adjoint solver.
    jlcase : Any
        Julia case object (needed to read PERMX/Y/Z values when log-scaling).
    parameter : str
        Parameter name. May contain ``"poro"``, ``"perm"``, ``"permx"``,
        ``"permy"``, ``"permz"`` (case-insensitive). Including ``"log"``
        enables log-scaling for permeability.
    actnum : np.ndarray
        Flat ACTNUM vector (1/0) for embedding into the full grid.
    perm_copied : bool
        If ``True``, treat PERMX as the master and sum contributions from all
        three permeability directions into the returned gradient.
    julia : juliacall.Main
        Julia main module.

    Returns
    -------
    np.ndarray
        Full-grid gradient vector.

    Raises
    ------
    ValueError
        If the gradient could not be located or the parameter is unsupported.
    """
    p = parameter.lower()

    # ---- Porosity -------------------------------------------------------- #
    if "poro" in p:
        grad = _extract_key_value(jlgrad, "porosity", julia)
        if grad is None:
            raise ValueError(f"Could not find porosity gradient for '{parameter}'")
        return _active_to_full_grid(np.asarray(grad), actnum)

    # ---- Permeability ---------------------------------------------------- #
    if "perm" in p:
        log_scale = "log" in p
        grad = _extract_key_value(jlgrad, "permeability", julia)
        if grad is None:
            raise ValueError(f"Could not find permeability gradient for '{parameter}'")

        full = np.asarray(grad)              # shape: (3, n_active)
        mdarcy = julia.seval("si_unit(:milli)*si_unit(:darcy)")  # mD → SI factor

        def _scale(adj: np.ndarray, axis: int) -> np.ndarray:
            """Apply log-scale (∂J/∂log(k) = k·∂J/∂k) or unit conversion."""
            if log_scale:
                perm = np.array(jlcase.input_data["GRID"][PERM_KEYS[axis]])
                return adj * perm.flatten(order="F")
            return adj * mdarcy

        # Copied-permeability case: aggregate gradients from all 3 axes.
        if perm_copied:
            out = np.zeros(actnum.shape, dtype=np.float64)
            for i in range(3):
                adj = _active_to_full_grid(full[i], actnum)
                out += _scale(adj, i)
            return out

        # Otherwise pick the axis encoded in the parameter name.
        idx = next((v for k, v in PERM_INDEX.items() if k in p), None)
        if idx is None:
            raise ValueError(f"Adjoint not implemented for '{parameter}'")
        return _scale(_active_to_full_grid(full[idx], actnum), idx)

    raise ValueError(f"Adjoint not implemented for parameter '{parameter}'")


def well_QOI_objective(wellID: str, phaseID: str, time: Iterable[float],
                       step_index=None, is_rate: bool = True, julia=None):
    """
    Build per-timestep Julia QOI closures for a single well/phase.

    Each closure returns either a daily-volume contribution (``is_rate=True``,
    one time only) or the cumulative integrand over time up to a horizon
    (``is_rate=False``). The sign is flipped automatically for producers so
    objectives are positive for "good" outcomes (e.g. produced oil).

    Parameters
    ----------
    wellID : str
        Name of the well as it appears in the case.
    phaseID : str
        Phase identifier (see :data:`RATE_ID_MAP`).
    time : iterable of float
        Sequence of horizon times in seconds.
    step_index : Any, optional
        Currently unused; retained for backwards compatibility.
    is_rate : bool, optional
        Selects rate (instantaneous) vs cumulative objective formulation.
    julia : juliacall.Main, optional
        Existing Julia module; one is imported on demand if omitted.

    Returns
    -------
    list[Any]
        List of Julia closures, one per entry in ``time``.

    Raises
    ------
    ValueError
        If ``phaseID`` is not in :data:`RATE_ID_MAP`.
    """
    if julia is None:
        from juliacall import Main as julia
        julia.seval("using JutulDarcy")

    if phaseID not in RATE_ID_MAP:
        raise ValueError(f"Unknown rate type: {phaseID}")
    rate_sym = RATE_ID_MAP[phaseID]

    qois = []
    for i, sec in enumerate(time):
        if is_rate:
            # Instantaneous rate: contribute only at the exact step matching `sec`.
            obj = julia.seval(
                f"""
                function well_QOI_{i}(model, state, dt, step_info, forces)
                    if step_info[:time] != {sec}
                        return 0.0
                    end
                    ctrl = forces[:Facility].control[Symbol("{wellID}")]
                    sign = ctrl isa JutulDarcy.ProducerControl ? -1.0 : 1.0
                    rate = JutulDarcy.compute_well_qoi(model, state, forces, Symbol("{wellID}"), {rate_sym})
                    return sign * rate * si_unit(:day) # rate is in Sm3/s: convert to Sm3/day
                end
                """
            )
        else:
            # Cumulative: integrate (dt * rate) up to the horizon `sec`.
            obj = julia.seval(
                f"""
                function well_QOI_{i}(model, state, dt, step_info, forces)
                    if step_info[:time] > {sec}
                        return 0.0
                    end
                    ctrl = forces[:Facility].control[Symbol("{wellID}")]
                    sign = ctrl isa JutulDarcy.ProducerControl ? -1.0 : 1.0
                    rate = JutulDarcy.compute_well_qoi(model, state, forces, Symbol("{wellID}"), {rate_sym})
                    return sign * dt * rate
                end
                """
            )
        qois.append(obj)
    return qois


# ============================================================================ #
# Main wrapper
# ============================================================================ #
class JutulDarcy:
    """
    Python wrapper around a JutulDarcy reservoir simulation.

    The class is instantiated once with an options dictionary describing the
    simulation setup, then called like a function on either a single input
    (``dict`` or ``.DATA`` path) or a list of inputs (one per ensemble member).

    See the module docstring for the list of supported options.

    Parameters
    ----------
    options : dict
        Configuration dictionary. Recognised keys:

        - ``runfile`` : str
            Path to a ``.mako`` template or ``.DATA`` file.
        - ``reporttype`` : {"days", "dates"}
            How report points are interpreted. Default ``"days"``.
        - ``reportpoint`` : list
            Report points (numeric days or :class:`datetime.datetime`).
        - ``datatype`` : list[str]
            Summary keywords to extract. Default
            ``["FOPT", "FGPT", "FWPT", "FWIT"]``.
        - ``adjoints`` : dict
            Objective/parameter spec; enables adjoint computation.
        - ``output_format`` : {"list", "dict", "dataframe"}
            Output container type. Default ``"dataframe"``.
        - ``adjoint_pbar`` : bool
            Show a per-objective progress bar. Default ``False``.
        - ``parallel`` : int
            Number of worker processes. Default ``1``.
        - ``perm_copied`` : bool
            See :func:`_extract_adjoint`. Default ``False``.
        - ``adjoint_mode`` : {"sensitivities", "optimization"}
            Selects the JutulDarcy gradient pathway. Default
            ``"sensitivities"``.
        - ``optimization_targets`` : Any
            Reserved for future use.
        - ``eval_adjoint_funcs`` : bool
            If True, store objective function values in ``self.adjoint_funcs``.

    Raises
    ------
    ValueError
        If ``reporttype``, ``output_format`` or ``adjoint_mode`` is invalid.
    """

    def __init__(self, options: dict):
        # ---- Runfile (template vs static deck) --------------------------- #
        runfile = options.get("runfile")
        self.makofile = runfile if runfile and runfile.endswith(".mako") else None
        self.datafile = runfile if runfile and runfile.endswith(".DATA") else None

        # ---- Report-point configuration ---------------------------------- #
        self.report_type = options.get("reporttype", "days")
        if self.report_type not in VALID_REPORT_TYPES:
            raise ValueError(
                f"Invalid reporttype '{self.report_type}'. "
                f"Must be one of {sorted(VALID_REPORT_TYPES)}"
            )
        self.report = options.get("reportpoint")
        self.index: list = [self.report_type, self.report]

        # Computed lazily during the first forward run.
        self.report_seconds: np.ndarray | None = None
        self.start_date: dt.datetime | None = None

        # ---- Datatypes to extract ---------------------------------------- #
        self.datatype = _process_datatype_info(
            options.get("datatype", ["FOPT", "FGPT", "FWPT", "FWIT"])
        )

        # ---- Adjoint configuration --------------------------------------- #
        if "adjoints" in options:
            self.compute_adjoints = True
            self.adjoint_info = _process_adjoint_info(options["adjoints"])
            self.eval_adjoint_funcs = options.get("eval_adjoint_funcs", False)
        else:
            self.compute_adjoints = False
            self.eval_adjoint_funcs = False
        self.adjoint_funcs: pd.DataFrame | None = None

        # ---- Output / execution options ---------------------------------- #
        self.output_format = options.get("output_format", "dataframe")
        if self.output_format not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output_format '{self.output_format}'. "
                f"Must be one of {sorted(VALID_OUTPUT_FORMATS)}"
            )
        self.adjoint_pbar = options.get("adjoint_pbar", False)
        self.parallel = int(options.get("parallel", 1))
        self.perm_copied = options.get("perm_copied", False)
        self.adjoint_mode = options.get("adjoint_mode", "sensitivities")
        if self.adjoint_mode not in VALID_ADJOINT_MODES:
            raise ValueError(
                f"Invalid adjoint_mode '{self.adjoint_mode}'. "
                f"Must be one of {sorted(VALID_ADJOINT_MODES)}"
            )
        self.optimization_targets = options.get("optimization_targets")

        # ---- PET compatibility attributes -------------------------------- #
        # `input_dict` and `true_order` are consumed by the surrounding PET
        # framework; we keep them as direct passthroughs.
        self.input_dict = options
        self.true_order = self.index

    # ---------------------------------------------------------------- #
    # Public entry point
    # ---------------------------------------------------------------- #
    def __call__(self, inputs: list[dict] | dict | str):
        """
        Run the configured simulation for one or many ensemble members.

        Parameters
        ----------
        inputs : list[dict] or dict or str
            Either a list of inputs (one per ensemble member), a single dict
            of Mako template parameters, or a path to a ``.DATA`` deck.

        Returns
        -------
        Forward-only mode
            Single input → one result; list input → list of results. The
            result type follows ``self.output_format``.
        Adjoint mode
            Same shape as above, but each result is a ``(forward, adjoint)``
            tuple where ``adjoint`` is a :class:`pandas.DataFrame` with a
            ``(objective, parameter)`` MultiIndex on the columns.
        """
        # Normalise to a list so the parallel path is uniform.
        if isinstance(inputs, (dict, str)):
            inputs = [inputs]

        # Clean up any stale simulation folders from a previous (possibly failed) run.
        self._cleanup_simulation_folders()

        n = len(inputs)
        outputs = p_map(
            self.run_fwd_sim,
            list(inputs),
            list(range(n)),
            num_cpus=self.parallel,
            unit="sim",
            desc="Simulations",
            leave=False,
            **PBAR_OPTS,
        )

        # In adjoint mode each worker returns a (result, adjoint) tuple; split
        # them out so callers see two parallel collections rather than a list
        # of tuples.
        if self.compute_adjoints:
            results, adjoints = zip(*outputs)
            if n == 1:
                return results[0], adjoints[0]
            return list(results), list(adjoints)

        return outputs[0] if n == 1 else outputs

    # ---------------------------------------------------------------- #
    # Per-member forward simulation
    # ---------------------------------------------------------------- #
    def run_fwd_sim(self, input: dict | str, idn: int = 0,
                    delete_folder: bool = True):
        """
        Run a single forward simulation (and optionally adjoints) in isolation.

        Each call creates and operates inside its own ``En_<idn>`` folder so
        multiple workers cannot collide on intermediate files written by
        JutulDarcy.

        Parameters
        ----------
        input : dict or str
            Mako template parameters or a path to a ``.DATA`` deck.
        idn : int, optional
            Ensemble member index (used for folder naming and pbar position).
        delete_folder : bool, optional
            If True (default), remove the ``En_<idn>`` folder when done.

        Returns
        -------
        Same as :meth:`__call__` for a single member.
        """
        # Julia is imported lazily inside the worker so it picks up the
        # process-local thread/state. Importing at module scope would break
        # multiprocessing's fork/spawn semantics.
        from juliacall import Main as julia
        julia.seval("using JutulDarcy, Jutul")

        folder = Path(f"En_{idn}")
        folder.mkdir(exist_ok=False)

        try:
            # Stage the deck (render template or copy file).
            datafile = self._stage_input(input, folder, idn)

            # All Julia file I/O must happen relative to the deck location.
            with _chdir(folder):
                case = self._setup_case(datafile, julia)
                julia.case = case  # expose to Julia-side eval'd expressions

                units = self._detect_units(case, julia)
                actnum_vec = self._extract_actnum(case)

                # Forward solve. `output_substates=True` keeps intermediate
                # states needed for adjoint reconstruction.
                jlres = julia.simulate_reservoir(
                    case, info_level=-1, output_substates=True
                )
                julia.res = jlres

                pyres = self.extract_datatypes(jlres, case, units, julia)
                output = self._format_output(pyres)

                if self.compute_adjoints:
                    adjoints = self._compute_adjoints(
                        case, jlres, pyres, units, actnum_vec, idn, julia
                    )
                    return output, adjoints

                return output
        finally:
            # Always clean up, even on failure, to keep the workspace tidy.
            if delete_folder and folder.exists():
                shutil.rmtree(folder, ignore_errors=True)

    # ---------------------------------------------------------------- #
    # Setup helpers
    # ---------------------------------------------------------------- #
    @staticmethod
    def _cleanup_simulation_folders() -> None:
        """Remove any ``En_*`` directories left in the current directory."""
        for item in os.listdir("."):
            if item.startswith("En_") and os.path.isdir(item):
                shutil.rmtree(item, ignore_errors=True)

    def _stage_input(self, input: dict | str, folder: Path, idn: int) -> str:
        """
        Render a Mako template or copy a static ``.DATA`` file into ``folder``.

        Parameters
        ----------
        input : dict or str
            Either Mako parameters or a path to a ``.DATA`` deck.
        folder : Path
            Destination directory.
        idn : int
            Ensemble index (injected into the Mako context as ``member``).

        Returns
        -------
        str
            Basename of the deck file inside ``folder``.

        Raises
        ------
        FileNotFoundError
            If a string input does not point to an existing file.
        ValueError
            If a string input is not a ``.DATA`` file.
        TypeError
            For unsupported input types.
        """
        if isinstance(input, dict):
            input["member"] = idn
            return self.render_makofile(self.makofile, str(folder), input)

        if isinstance(input, str):
            if not os.path.isfile(input):
                raise FileNotFoundError(f"Input file {input} not found")
            if not input.endswith(".DATA"):
                raise ValueError("Input string must be a path to a .DATA file")
            datafile = os.path.basename(input)
            shutil.copy(input, folder)
            return datafile

        raise TypeError(f"Unsupported input type: {type(input).__name__}")

    @staticmethod
    def _setup_case(datafile: str, julia):
        """Invoke JutulDarcy's case-setup routine with output suppressed."""
        return _suppress_julia(julia, f'setup_case_from_data_file("{datafile}")')

    @staticmethod
    def _detect_units(case, julia) -> str | Any:
        """
        Detect the unit system from the deck's RUNSPEC section.

        Returns
        -------
        str
            One of ``"metric"``, ``"si"``, ``"field"`` if specified.
        Any
            ``julia.missing`` if no unit keyword is present.
        """
        runspec = case.input_data["RUNSPEC"]
        for key, name in (("METRIC", "metric"), ("SI", "si"), ("FIELD", "field")):
            if julia.haskey(runspec, key):
                return name
        return julia.missing

    @staticmethod
    def _extract_actnum(case) -> np.ndarray:
        """
        Return the flat ACTNUM array (1/0), defaulting to all-active.

        Parameters
        ----------
        case : Any
            Julia case object.

        Returns
        -------
        np.ndarray
            Flat (Fortran-ordered) ACTNUM vector.
        """
        nx, ny, nz = case.input_data["GRID"]["cartDims"]
        try:
            actnum = np.array(case.input_data["GRID"]["ACTNUM"])
            return actnum.flatten(order="F")
        except (KeyError, AttributeError):
            # No ACTNUM keyword in the deck => every cell is active.
            return np.ones(nx * ny * nz)

    def _format_output(self, pyres: pd.DataFrame):
        """Convert the per-member DataFrame to the user-requested container."""
        if self.output_format == "dataframe":
            return pyres
        if self.output_format == "dict":
            return pyres.to_dict(orient="list")
        return pyres.to_dict(orient="records")  # 'list' format → list of records

    @staticmethod
    def render_makofile(makofile: str, folder: str, input: dict) -> str:
        """
        Render a Mako template into a ``.DATA`` file in ``folder``.

        Parameters
        ----------
        makofile : str
            Path to the ``.mako`` template.
        folder : str
            Destination directory.
        input : dict
            Template context variables.

        Returns
        -------
        str
            Basename of the rendered ``.DATA`` file.
        """
        datafile = os.path.basename(makofile).replace(".mako", ".DATA")
        template = Template(filename=makofile)
        with open(os.path.join(folder, datafile), "w") as f:
            f.write(template.render(**input))
        return datafile

    # ---------------------------------------------------------------- #
    # Datatype extraction
    # ---------------------------------------------------------------- #
    def extract_datatypes(self, jlres, jlcase, units, julia) -> pd.DataFrame:
        """
        Extract requested summary keywords from a finished simulation.

        Parameters
        ----------
        jlres : Any
            Julia result object from ``simulate_reservoir``.
        jlcase : Any
            Julia case object (used for the start date).
        units : str or Any
            Unit-system identifier (see :meth:`_detect_units`).
        julia : juliacall.Main
            Julia main module.

        Returns
        -------
        pd.DataFrame
            One row per report point; one column per datatype. ``df.attrs``
            holds per-column unit strings.

        Raises
        ------
        ValueError
            If a requested datatype or well is missing from the results.
        """
        jl_units = julia.Symbol(units) if isinstance(units, str) else units
        smry = julia.JutulDarcy.summary_result(jlcase, jlres, jl_units)

        # JutulDarcy reports time in integer seconds; match against our
        # requested report points by intersecting the two integer arrays.
        sim_seconds = np.array(list(smry["TIME"].seconds), dtype=np.int64)
        self.start_date = jlcase.input_data["RUNSPEC"]["START"]
        self.report_seconds = self._compute_report_seconds()

        idx = np.flatnonzero(np.isin(sim_seconds, self.report_seconds))

        res: dict[str, np.ndarray] = {}
        attrs: dict[str, str] = {}

        wells = smry["VALUES"]["WELLS"]
        field = smry["VALUES"]["FIELD"]

        for datatype in self.datatype:
            if ":" in datatype:
                # Well-level datatype: "<baseID>:<wellID>"
                baseID, wellID = datatype.split(":")
                if wellID not in wells:
                    raise ValueError(
                        f"Well ID '{wellID}' not found for datatype '{baseID}'"
                    )
                well_data = wells[wellID]
                if baseID not in well_data:
                    raise ValueError(
                        f"Datatype '{baseID}' not found for well '{wellID}'"
                    )
                res[datatype] = np.array(well_data[baseID])[idx]
                attrs[datatype] = get_metric_unit(baseID)
            else:
                # Field-level datatype.
                if datatype not in field:
                    raise ValueError(
                        f"Datatype '{datatype}' not found in field results"
                    )
                res[datatype] = np.array(field[datatype])[idx]
                attrs[datatype] = get_metric_unit(datatype)

        df = pd.DataFrame(res, index=self.index[1])
        df.index.name = self.index[0]
        df.attrs = attrs
        return df

    def _compute_report_seconds(self) -> np.ndarray:
        """
        Convert the configured report points to integer seconds.

        Returns
        -------
        np.ndarray
            Seconds elapsed from the simulation start for each report point.

        Raises
        ------
        ValueError
            If ``self.report_type`` is not recognised (defensive; should be
            blocked by :meth:`__init__` validation).
        """
        rtype, rpoints = self.index
        if rtype == "days":
            return np.array(rpoints, dtype=np.int64) * SECONDS_PER_DAY
        if rtype == "dates":
            return np.array(
                [(d - self.start_date).total_seconds() for d in rpoints],
                dtype=np.int64,
            )
        raise ValueError(f"Invalid report type: {rtype}")

    # ---------------------------------------------------------------- #
    # Adjoint computation
    # ---------------------------------------------------------------- #
    def _compute_adjoints(self, case, jlres, pyres, units, actnum_vec,
                          idn: int, julia) -> pd.DataFrame:
        """
        Compute gradients for all configured adjoint objectives.

        Parameters
        ----------
        case : Any
            Julia case object.
        jlres : Any
            Julia simulation result.
        pyres : pd.DataFrame
            Forward-extracted results, used as a sanity check against
            Jutul's own objective evaluation.
        units : Any
            Unit system identifier (stored in returned ``df.attrs``).
        actnum_vec : np.ndarray
            Flat ACTNUM vector for full-grid embedding.
        idn : int
            Ensemble index (used only for pbar positioning).
        julia : juliacall.Main
            Julia main module.

        Returns
        -------
        pd.DataFrame
            Index: requested adjoint evaluation points;
            columns: ``(objective, parameter)`` MultiIndex.
        """
        # Optimization mode requires setting up a parameter dictionary and
        # freeing the parameters Jutul will differentiate w.r.t.
        if self.adjoint_mode == "optimization":
            julia.grad_case = julia.seval(
                "JutulDarcy.setup_reservoir_dict_optimization(case, verbose=false)"
            )
            julia.seval("JutulDarcy.free_optimization_parameters!(grad_case)")

        # Pre-allocate output containers.
        grad_dict: dict[tuple[str, str], list] = {
            (col, p): []
            for col, obj in self.adjoint_info.items()
            for p in obj.parameters
        }
        func_dict: dict[str, list] = (
            {col: [] for col in self.adjoint_info} if self.eval_adjoint_funcs else {}
        )

        pbar = self._make_adjoint_pbar(idn)
        sim_times = np.array(jlres.time) # Unit: seconds
        adjoint_index_final = None  # captured from the last objective

        for col, info in pbar:
            # Resolve the evaluation points for this objective.
            adj_seconds, adj_index = self._resolve_adjoint_steps(info)
            adjoint_index_final = adj_index

            # Closest simulation step to each requested time (Julia is 1-indexed).
            adj_step_idx = [
                int(np.argmin(np.abs(sim_times - s))) + 1 for s in adj_seconds
            ]

            funcs = well_QOI_objective(
                info.wellID, info.phase, adj_seconds,
                adj_step_idx, info.is_rate, julia=julia,
            )

            if self.adjoint_pbar:
                pbar.set_description_str(f"Adjoints for {col}")

            for i, func in enumerate(funcs):
                julia.func = func
                grad = self._solve_adjoint(julia)

                # Cross-check Julia's objective evaluation against the value
                # we already obtained from the forward extraction.
                func_val = julia.Jutul.evaluate_objective(func, case, jlres.result)
                expected = pyres.loc[adj_index[i]][col]
                assert np.isclose(func_val, expected), (
                    f"func_val: {func_val:.3e}, pyres: {expected:.3e}"
                )
                if self.eval_adjoint_funcs:
                    func_dict[col].append(func_val)

                # Post-process the raw Julia gradient for each requested parameter.
                for param in info.parameters:
                    grad_dict[(col, param)].append(
                        _extract_adjoint(grad, case, param, actnum_vec,
                                         self.perm_copied, julia)
                    )

        if self.adjoint_pbar:
            pbar.close()

        # Wrap the gradients into a multi-indexed DataFrame.
        cols = pd.MultiIndex.from_tuples(grad_dict.keys())
        adjoints = pd.DataFrame(grad_dict, columns=cols, index=adjoint_index_final)
        adjoints.index.name = self.index[0]
        adjoints.attrs = {"units": units}

        if self.eval_adjoint_funcs:
            fun = pd.DataFrame(func_dict, index=adjoints.index)
            fun.index.name = adjoints.index.name
            self.adjoint_funcs = fun

        return adjoints

    def _solve_adjoint(self, julia):
        """
        Invoke the appropriate JutulDarcy adjoint solver for the current mode.

        Parameters
        ----------
        julia : juliacall.Main
            Julia main module. The names ``case``, ``res``, ``grad_case`` and
            ``func`` are expected to be already bound in the Julia namespace.

        Returns
        -------
        Any
            Raw Julia gradient object.
        """
        if self.adjoint_mode == "sensitivities":
            return _suppress_julia(
                julia,
                "JutulDarcy.reservoir_sensitivities("
                "case, res, func, include_parameters=true)"
            )
        return _suppress_julia(
            julia,
            "JutulDarcy.parameters_gradient_reservoir(grad_case, func, deps=:case)"
        )

    def _resolve_adjoint_steps(self, info: AdjointObjective):
        """
        Convert an objective's step specification into seconds + index labels.

        Parameters
        ----------
        info : AdjointObjective
            Spec containing ``steps`` (``"all"``, list of ints, or list of
            :class:`datetime.datetime`).

        Returns
        -------
        tuple
            ``(seconds_array, index_labels)`` ready for adjoint evaluation.

        Raises
        ------
        TypeError
            If the step list contains an unsupported element type.
        """
        if info.steps == "all":
            return self.report_seconds, self.index[1]

        first = info.steps[0]
        if isinstance(first, int):
            return (
                np.array(info.steps, dtype=np.int64) * SECONDS_PER_DAY,
                info.steps,
            )
        if isinstance(first, dt.datetime):
            return (
                np.array(
                    [(d - self.start_date).total_seconds() for d in info.steps],
                    dtype=np.int64,
                ),
                info.steps,
            )
        raise TypeError(f"Unsupported adjoint step type: {type(first).__name__}")

    def _make_adjoint_pbar(self, idn: int):
        """
        Build an iterator over ``self.adjoint_info``, optionally with a pbar.

        Parameters
        ----------
        idn : int
            Ensemble index (used to stagger pbar positions in parallel runs).

        Returns
        -------
        Iterable
            Either ``self.adjoint_info.items()`` or a wrapped :class:`tqdm`.
        """
        if not self.adjoint_pbar:
            return self.adjoint_info.items()

        # Drop the default colour so we can override it for the adjoint bars.
        opts = {k: v for k, v in PBAR_OPTS.items() if k != "colour"}
        return tqdm(
            self.adjoint_info.items(),
            desc="Solving adjoints",
            position=idn + 1,
            leave=False,
            unit="obj",
            dynamic_ncols=False,
            colour="#713996",
            **opts,
        )