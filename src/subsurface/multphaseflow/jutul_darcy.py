'''
Simulator wrapper for the JutulDarcy simulator.

This module provides a wrapper interface for running JutulDarcy simulations
with support for ensemble-based workflows and flexible output formatting.
'''

import os
import shutil
import warnings

import numpy as np
import pandas as pd
import datetime as dt

from mako.template import Template
from p_tqdm import p_map
from tqdm import tqdm


__author__ = "Mathias Methlie Nilsen"
__all__ = ["JutulDarcy"]


os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
os.environ["PYTHON_JULIACALL_THREADS"] = "1"
os.environ["PYTHON_JULIACALL_OPTLEVEL"] = "3"
warnings.filterwarnings("ignore", message=".*juliacall module already imported.*")


PBAR_OPTS = {
    "ncols": 110,
    "colour": "#285475",
    "bar_format": "{desc}: {percentage:3.0f}% [{bar}] {n_fmt}/{total_fmt} │ ⏱ {elapsed}<{remaining} │ {rate_fmt}",
    "ascii": "-◼",
}


SECONDS_PER_DAY = 24 * 60 * 60
CUMULATIVE_KEYS = {
    "FOPT", "FGPT", "FWPT", "FWLT", "FWIT",
    "WOPT", "WGPT", "WWPT", "WLPT",
}


class JutulDarcy:

    def __init__(self, options: dict):
        """
        JutulDarcy simulation wrapper.

        Parameters
        ----------
        options : dict
            Configuration dictionary controlling input files, reporting, output
            formatting, parallel execution, and optional adjoint computation.
            Supported keys:

            - ``runfile`` : str, optional
                Path to either a ``.mako`` template or a ``.DATA`` run file.

            - ``reporttype`` : {"days", "dates"}, optional
                Type of report index. Default is ``"days"``.

            - ``reportpoint`` : list, optional
                Report points used for output indexing. For ``"days"`` this is
                a list of numeric day values; for ``"dates"`` a list of
                ``datetime`` objects.

            - ``datatype`` : list[str], optional
                Result keywords to extract. Supports field keys (for example
                ``"FOPT"``) and well keys (for example ``"WOPR:PROD1"``).

            - ``adjoints`` : dict, optional
                Objective and parameter configuration for adjoint sensitivities.

            - ``output_format`` : {"list", "dict", "dataframe"}, optional
                Output representation for forward results. Default is ``"list"``. 
                (will probably be dataframe in the future)

            - ``adjoint_pbar`` : bool, optional
                If ``True``, display progress bars during adjoint solves. 
                Not recommended for large ensembles due to many progress bars. 
                Default is ``True``.

            - ``parallel`` : int, optional
                Number of processes for ensemble simulations. Default is ``1``.
        """
        # Check runfile
        runfile = options.get('runfile')
        if runfile:
            self.makofile = runfile if runfile.endswith('.mako') else None
            self.datafile = runfile if runfile.endswith('.DATA') else None

        # Report Options 
        self.report_type = options.get('reporttype', 'days') # days or dates
        self.report = options.get('reportpoint', None)       # list of days or dates (dates as datetime objects)
        self.index = [self.report_type, self.report]
        self.report_seconds = None
        self.start_date = None

        # Process datatypes
        datatype = options.get('datatype', ['FOPT', 'FGPT', 'FWPT', 'FWIT'])
        self.datatype = _process_datatype_info(datatype)

        # Adjoint information
        if 'adjoints' in options:
            self.compute_adjoints = True
            self.adjoint_info = _process_adjoint_info(options['adjoints']) 
            self.eval_adjoint_funcs = options.get('eval_adjoint_funcs', False)
            self.adjoint_funcs = None
        else:
            self.compute_adjoints = False
            self.eval_adjoint_funcs = False
            self.adjoint_funcs = None

        # Other options
        self.output_format = options.get('output_format', 'dataframe') # list, dict or dataframe
        self.adjoint_pbar = options.get('adjoint_pbar', True)
        self.parallel = options.get('parallel', 1)

        # This is for PET to work properly
        self.input_dict = options
        self.true_order = self.index

    
    def __call__(self, inputs: list[dict]|dict|str):
        """
        Execute parallel forward simulations for all ensemble members.

        This method runs the forward simulation for all input parameter sets in parallel,
        optionally computing adjoint sensitivities if configured.

        Parameters
        ----------
        inputs : list of dict, dict, or str
            List containing input parameter sets, indexed by ensemble member, or a single input dictionary,
            or a string representing a file path to input datafile.

            number (0, 1, 2, ...). Each value is a dict of parameters for that member.

        Returns
        -------
        output : list of (dict, list, or pd.DataFrame)
            Forward simulation results for all ensemble members, formatted according
            to the 'out_format' option specified during initialization.
        adjoints : pd.DataFrame, optional
            Adjoint sensitivities for all objectives and parameters. Only returned if
            'adjoints' configuration was provided during initialization. Contains
            multi-indexed columns (objective, parameter) and index based on reporttype.

        Notes
        -----
        - Existing simulation folders (En_*) are deleted before execution to prevent
          conflicts from previous runs.
        - Parallel execution uses the number of CPUs specified in the 'parallel' option.
        - Progress is displayed via a progress bar during execution.

        Examples
        --------
        >>> simulator = JutulDarcyWrapper(options)
        >>> inputs = [{'param1': [1, 2, 3]}, {'param1': [1.1, 2.1, 3.1]}]
        >>> results = simulator(inputs)
        """
        if isinstance(inputs, (dict, str)):
            inputs = [inputs]

        # Delet all existing En_* folders
        for item in os.listdir('.'):
            if os.path.isdir(item) and item.startswith('En_'):
                shutil.rmtree(item)
        
        # simulate all inputs in parallel
        outputs = p_map(
            self.run_fwd_sim, 
            [inputs[n] for n in range(len(inputs))], 
            list(range(len(inputs))), 
            num_cpus=self.parallel,
            unit='sim',
            desc='Simulations',
            leave=False,
            **PBAR_OPTS
        )

        if self.compute_adjoints:
            results, adjoints = zip(*outputs)
            if len(inputs) == 1:
                results  = results[0]
                adjoints = adjoints[0]

            return results, adjoints
        else:
            return outputs
                     

    def run_fwd_sim(self, input: dict|str, idn: int=0, delete_folder: bool=True):

        # Import Julia and JutulDarcy (this needs to be done here for multiprocessing to work properly)
        from juliacall import Main as julia
        julia.seval('using JutulDarcy, Jutul')
        
        # Make simulation folder
        folder = f'En_{idn}'
        os.makedirs(folder)

        # Check input type (datafile or input for makofile)
        if isinstance(input, dict):
            input['member'] = idn
            datafile = self.render_makofile(self.makofile, folder, input)
        elif isinstance(input, str):
            assert os.path.isfile(input), f'Input file {input} not found'
            assert input.endswith('.DATA'), 'Input string must be a path to a .DATA file'
            
            # Copy datafile to simulation folder
            datafile = os.path.basename(input)
            shutil.copy(input, folder)

        # Enter simulation folder and run simulation
        os.chdir(folder)

        # Setup case from datafile
        suppress = True
        if suppress:
            case = julia.seval(f"""
            redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                    setup_case_from_data_file("{datafile}")
                end
            end
            """)
        else:
            case = julia.setup_case_from_data_file(datafile)

        # Get Units
        if julia.haskey(case.input_data["RUNSPEC"], "METRIC"):
            units = 'metric'
        elif julia.haskey(case.input_data["RUNSPEC"], "SI"):
            units = 'si'
        elif julia.haskey(case.input_data["RUNSPEC"], "FIELD"):
            units = 'field'
        else:
            units = julia.missing

        # Get some grid info
        nx, ny, nz = case.input_data["GRID"]["cartDims"]
        try:
            actnum = np.array(case.input_data["GRID"]["ACTNUM"]) # Shape (nx, ny, nz)
            actnum_vec = actnum.flatten(order='F')  # Fortran order flattening
        except:
            actnum_vec = np.ones(nx*ny*nz)

        # Simulate and get results
        jlres = julia.simulate_reservoir(case, info_level=-1, output_substates=True)
        pyres = self.extract_datatypes(jlres, case, units, julia)
        
        # Convert output to requested format
        if not self.output_format == 'dataframe':
            if self.output_format == 'dict':
                output = pyres.to_dict(orient='list')
            elif self.output_format == 'list':
                output = pyres.to_dict(orient='records')
        else:
            output = pyres


        # ----------------------------------------------------------------------------------------------
        # Compute adjoints
        # ----------------------------------------------------------------------------------------------
        if self.compute_adjoints:

            # Initialize adjoint results storage
            columns = self.adjoint_info.keys()
            grad_dict = {(col, param): [] for col in columns for param in self.adjoint_info[col]['parameters']}

            if self.eval_adjoint_funcs:
                func_dict = {col: [] for col in columns} # For storing functions for each objective

            # Setup progress bar (iterate over adjoint objectives, not DataFrame columns)
            if self.adjoint_pbar:
                PBAR_OPTS.pop('colour', None)
                pbar = tqdm(
                    self.adjoint_info.items(), 
                    desc=f'Solving adjoints',
                    position=idn+1,
                    leave=False,
                    unit='obj',
                    dynamic_ncols=True,
                    colour="#713996",
                    **PBAR_OPTS
                )
            else:
                pbar = self.adjoint_info.items()


            # Loop over adjoint objectives
            for col, info in pbar:

                if info['steps'] == 'all': # If 'all', use same steps as in report points
                    adjoint_index = self.index[1]
                    adjoint_seconds = self.report_seconds
                else:
                    # DAYS
                    if isinstance(info['steps'][0], int):
                        adjoint_seconds = np.array(info['steps'], dtype=np.int64) * SECONDS_PER_DAY
                        adjoint_index = info['steps']

                    # DATES
                    elif isinstance(info['steps'][0], dt.datetime):
                        adjoint_seconds = np.array([(d - self.start_date).total_seconds() for d in info['steps']], dtype=np.int64)
                        adjoint_index = info['steps']

                # Map each time in adjoint_seconds to the closest step in the simulation output (jlres.time)
                adjoint_step_idx = [int(np.argmin(np.abs(np.array(jlres.time) - s))) + 1 for s in adjoint_seconds]

                # Get QOI function for this objective
                funcs = well_QOI_objective(
                    info['wellID'], 
                    info['phase'], 
                    adjoint_seconds,
                    adjoint_step_idx,
                    info['is_rate'], 
                    julia=julia
                )

                # Update progress bar
                if self.adjoint_pbar:
                    update_desc = f'Adjoints for {col}'
                    pbar.set_description_str(update_desc)

                # Compute adjoint sensitivities
                julia.case = case
                julia.res = jlres
                for i, func in enumerate(funcs):
                    julia.func = func
                    grad = julia.seval("""
                    redirect_stdout(devnull) do
                        redirect_stderr(devnull) do
                            JutulDarcy.reservoir_sensitivities(
                                case, res, func,
                                include_parameters=true,           
                            )
                        end
                    end
                    """)

                    # Evaluate function value for this objective at this time step and store in dict (for checking)
                    func_val = None
                    double_check = True
                    if double_check:
                        func_val = julia.Jutul.evaluate_objective(func, case, jlres.result)
                        msg = f'func_val: {func_val:.3e}, pyres: {pyres.loc[adjoint_index[i]][col]:.3e}'
                        assert np.isclose(func_val, pyres.loc[adjoint_index[i]][col]), msg
                    
                    if self.eval_adjoint_funcs:
                        if func_val is not None:
                            func_dict[col].append(func_val)
                        else:
                            func_val = julia.Jutul.evaluate_objective(func, case, jlres.result)
                            func_dict[col].append(func_val)


                    # Extract parameter sensitivities for this objective and store in dict
                    for param in info['parameters']:
                        grad_param = _extract_adjoint(grad, case, param, actnum_vec, info['is_rate'], julia)
                        grad_dict[(col, param)].append(grad_param)

            if self.adjoint_pbar:
                pbar.close()

            # Create adjoint dataframe with MultiIndex columns
            cols = pd.MultiIndex.from_tuples(grad_dict.keys())
            adjoints = pd.DataFrame(grad_dict, columns=cols, index=adjoint_index)
            adjoints.index.name = self.index[0]
            adjoints.attrs = {'units': units}
        # ----------------------------------------------------------------------------------------------

        os.chdir('..')

        # Function values dataframe
        if self.eval_adjoint_funcs:
            fun = pd.DataFrame(func_dict, index=adjoints.index)
            fun.index.name = adjoints.index.name
            self.adjoint_funcs = fun
        
        # Delete simulation folder
        if delete_folder:
            shutil.rmtree(folder)
        
        if self.compute_adjoints:
            return output, adjoints
        else:
            return output


    
    def extract_datatypes(self, jlres, jlcase, units, julia) -> tuple[pd.DataFrame, np.ndarray]:
        res = {}
        attrs = {} # For datatype units

        if isinstance(units, str):
            jl_units = julia.Symbol(units)
        else:
            jl_units = units  # julia.missing or already a Julia Symbol

        # Summary
        smry = julia.JutulDarcy.summary_result(jlcase, jlres, jl_units)

        # Reportponts in seconds
        sim_seconds = np.array(list(smry["TIME"].seconds), dtype=np.int64)
        self.start_date = jlcase.input_data["RUNSPEC"]['START']

        if self.index[0] == 'days':
            self.report_seconds = np.array(self.index[1], dtype=np.int64) * SECONDS_PER_DAY
        elif self.index[0] == 'dates':
            self.report_seconds = np.array([(d - self.start_date).total_seconds() for d in self.index[1]], dtype=np.int64)
        else:
            raise ValueError(f"Invalid report type: {self.index[0]}. Must be 'days' or 'dates'.")

        # Index of report points in simulation output
        idx = np.argwhere(np.isin(sim_seconds, self.report_seconds)).flatten()
    
        # Extract datatypes for step
        for datatype in self.datatype:  
            
            # Well results
            if ':' in datatype:
                baseID, wellID = datatype.split(':')
                jlres_wells = smry["VALUES"]["WELLS"]

                if wellID in jlres_wells:
                    data = jlres_wells[wellID]
                else:
                    raise ValueError(f"Well ID '{wellID}' not found in simulation results for datatype '{baseID}'")
                
                if baseID in data:
                    values = np.array(data[baseID])[idx]
                    res[datatype] = values
                    attrs[datatype] = get_metric_unit(baseID)
                else:
                    raise ValueError(f"Datatype '{baseID}' not found for well '{wellID}' in simulation results")
            
            # Field results
            else:
                jlres_field = smry["VALUES"]["FIELD"]
                if datatype in jlres_field:
                    data = np.array(jlres_field[datatype])
                    res[datatype] = data[idx]
                    attrs[datatype] = get_metric_unit(datatype)
                else:
                    raise ValueError(f"Datatype '{datatype}' not found in field results of simulation")
            
            # TODO: Add support for other datatypes (saturation, pressure, PERMX, etc.)
                
        # Make DataFrame
        res = pd.DataFrame(res, index=self.index[1])
        res.index.name = self.index[0]
        res.attrs = attrs

        return res

    
    def render_makofile(self, makofile: str, folder: str, input: dict):
        datafile = makofile.replace('.mako', '.DATA')
        template = Template(filename=makofile)
        with open(os.path.join(folder, datafile), 'w') as f:
            f.write(template.render(**input))
        
        return datafile


               
def _extract_adjoint(jlgrad, jlcase, parameter, actnum, is_rate, julia):
     
    if 'poro' in parameter.lower():
        adjoint = np.asarray(jlgrad[julia.Symbol("porosity")]) # F-order array (only active cells)
        adjoint = _active_to_full_grid(adjoint, actnum)

    elif 'perm' in parameter.lower():
        adjoint = np.asarray(jlgrad[julia.Symbol("permeability")]) # F-order array (only active cells)
        
        if 'permx' in parameter.lower(): index = 0
        if 'permy' in parameter.lower(): index = 1
        if 'permz' in parameter.lower(): index = 2
        
        adjoint = _active_to_full_grid(adjoint[index], actnum) # SI: per m2, convert to per mD

        if 'log' in parameter.lower():
            perm = np.array(jlcase.input_data['GRID'][['PERMX', 'PERMY', 'PERMZ'][index]])
            adjoint = adjoint * perm.flatten(order='F')
            # Note: For dJ/dlog(perm) = dJ/dperm * perm, no need to convert from m2 to mD.
        else:
            adjoint = adjoint * julia.seval('si_unit(:milli)*si_unit(:darcy)')
    else:
        raise ValueError(f"Adjoint not implemented for parameter '{parameter}'")

    return adjoint
        

def _active_to_full_grid(vec, actnum_vec, fill_value=0.0):
    if len(vec) == actnum_vec.sum():
        full_vec = np.full(actnum_vec.shape, fill_value, dtype=np.float64, order='F')
        full_vec[actnum_vec == 1] = vec
        return full_vec
    if len(vec) == len(actnum_vec):
        return vec.astype(np.float64, copy=False)

    raise ValueError("Parameter length does not match number of active cells")
        

def _process_datatype_info(datatypes):
    processed = []
    for dataID in datatypes:
        if ':' in dataID:
            s = dataID.split(':')
            baseID = s[0]
            wellIDs = s[1:]
            for wID in wellIDs:
                processed.append(f'{baseID}:{wID}')
        else:
            processed.append(dataID)
    return processed
    
def _process_adjoint_info(adjoint_info):
    phase_map = {
        'WOPT': ('oil', False),
        'WGPT': ('gas', False),
        'WWPT': ('water', False),
        'WLPT': ('liquid', False),
        'WOPR': ('oil', True),
        'WGPR': ('gas', True),
        'WWPR': ('water', True),
        'WWIR': ('water', True),
        'WLPR': ('liquid', True),
    }
    info = {}
    for dataID in adjoint_info:
        wellID = adjoint_info[dataID]['wellID']
        if isinstance(wellID, str): wellID = [wellID]

        parameters = adjoint_info[dataID]['parameters']
        if isinstance(parameters, str):
            parameters = [parameters]

        steps = adjoint_info[dataID]['steps'] # 'all' or 'acc' or list of steps (days or dates)
        phaseID, is_rate = phase_map[dataID]
        for wID in wellID:
            info[f'{dataID}:{wID}'] = {
                'wellID': wID,
                'phase': phaseID,
                'is_rate': is_rate,
                'parameters': parameters,
                'steps': steps,
            }
    return info
        
        
def get_metric_unit(key: str) -> str:
    unit_map = {
        "PORO": "",
        "PERMX": "mD",
        "PERMY": "mD",
        "PERMZ": "mD",
        "FOPT": "Sm3",
        "FGPT": "Sm3",
        "FWPT": "Sm3",
        "FWLT": "Sm3",
        "FWIT": "Sm3",
        "FOPR": "Sm3/day",
        "FGPR": "Sm3/day",
        "FWPR": "Sm3/day",
        "FLPR": "Sm3/day",
        "FWIR": "Sm3/day",
        "WOPR": "Sm3/day",
        "WGPR": "Sm3/day",
        "WWPR": "Sm3/day",
        "WLPR": "Sm3/day",
        "WWIR": "Sm3/day",
    }
    return unit_map.get(key.upper(), "Unknown")


def well_QOI_objective(wellID, phaseID, time=None, step_index=None, is_rate=True, julia=None):
    if julia is None:
        from juliacall import Main as julia
        julia.seval("using JutulDarcy")

    rateID_map = {
        "mass"   : "TotalSurfaceMassRate",
        "liquid" : "SurfaceLiquidRateTarget",
        "water"  : "SurfaceWaterRateTarget",
        "oil"    : "SurfaceOilRateTarget",
        "gas"    : "SurfaceGasRateTarget",
        "rate"   : "TotalRateTarget",
    }
    if phaseID not in rateID_map:
        raise ValueError(f"Unknown rate type: {phaseID}")
    rateID_symbol = rateID_map[phaseID]

    qois = []
    for i, sec in enumerate(time):
        if is_rate:
            obj = julia.seval(
                f"""
                function well_QOI_{i}(model, state, dt, step_info, forces)
                    if step_info[:time] != {sec}
                        return 0.0
                    end
                    ctrl = forces[:Facility].control[Symbol("{wellID}")]
                    sign = ctrl isa JutulDarcy.ProducerControl ? -1.0 : 1.0
                    rate = JutulDarcy.compute_well_qoi(model, state, forces, Symbol("{wellID}"), {rateID_symbol})
                    return sign * rate * si_unit(:day) # rate is in Sm3/s: convert to Sm3/day
                end
                """
            )
        else:
            obj = julia.seval(
                f"""
                function well_QOI_{i}(model, state, dt, step_info, forces)
                    if step_info[:time] > {sec}
                        return 0.0
                    end
                    ctrl = forces[:Facility].control[Symbol("{wellID}")]
                    sign = ctrl isa JutulDarcy.ProducerControl ? -1.0 : 1.0
                    rate = JutulDarcy.compute_well_qoi(model, state, forces, Symbol("{wellID}"), {rateID_symbol})
                    return sign * dt * rate
                end
                """
            )
        qois.append(obj)

    return qois
