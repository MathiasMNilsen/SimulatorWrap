'''
Simulator wrapper for the JutulDarcy simulator.

This module provides a wrapper interface for running JutulDarcy simulations
with support for ensemble-based workflows and flexible output formatting.
'''

#────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import warnings
import shutil
import os

from mako.template import Template
from typing import Union
from p_tqdm import p_map
from tqdm import tqdm
#────────────────────────────────────────────────────


__author__ = 'Mathias Methlie Nilsen'
__all__ = ['JutulDarcyWrapper']


#────────────────────────────────────────────────────────────────────────────────────
os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes'
os.environ['PYTHON_JULIACALL_THREADS'] = '1'
os.environ['PYTHON_JULIACALL_OPTLEVEL'] = '3'
warnings.filterwarnings('ignore', message='.*juliacall module already imported.*')
#────────────────────────────────────────────────────────────────────────────────────

PBAR_OPTS = {
    'ncols': 110,
    'colour': "#285475",
    'bar_format': '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    'ascii': '-◼', # Custom bar characters for a sleeker look
}


class JutulDarcyWrapper:

    def __init__(self, options):
        '''
        Wrapper for the JutulDarcy simulator [1].

        Parameters
        ----------
        options : dict
            Configuration options for the wrapper.
            Keys:
                - 'makofile' or 'runfile': Path to the makofile.mako or runfile.DATA template.
                - 'reporttype': Type of report (default: 'days').
                - 'out_format': Output format ('list', 'dict', 'dataframe'; default: 'list').
                - 'datatype': List of data types to extract (default: ['FOPT', 'FGPT', 'FWPT', 'FWIT']).
                - 'parallel': Number of parallel simulations (default: 1).

        References
        ----------
        [1] Møyner, O. (2025).
            JutulDarcy.jl - a fully differentiable high-performance reservoir simulator
            based on automatic differentiation. Computational Geosciences, 29, Article 30.
            https://doi.org/10.1007/s10596-025-10366-6
        '''
        # Make makofile an mandatory option
        if ('makofile' not in options) and ('runfile' not in options):
            raise ValueError('Wrapper  requires a makofile (or runfile) option')
        
        if 'makofile' in options: 
            self.makofile = options.get('makofile')

        if 'runfile' in options:
            self.makofile = options.get('runfile').split('.')[0] + '.mako'

        # Other variables
        self.reporttype = options.get('reporttype', 'days')
        self.out_format = options.get('out_format', 'list')
        self.datatype   = options.get('datatype', ['FOPT', 'FGPT', 'FWPT', 'FWIT'])
        self.parallel   = options.get('parallel', 1)
        self.units      = options.get('units', 'metric') # This is not consistently used!
        self.datafile = None
        self.compute_adjoints = False

        # Process datatypes
        if isinstance(self.datatype, list) and len(self.datatype) == 1 and isinstance(self.datatype[0], dict):
            d = []
            for key in self.datatype[0]:
                if self.datatype[0][key] is None:
                    d.append(key)
                else:
                    for well in self.datatype[0][key]:
                        d.append(f'{key}:{well}')
            self.datatype = d
        elif isinstance(self.datatype, list) and all(isinstance(dt, str) for dt in self.datatype):
            self.datatype = self.datatype
        else:
            assert self.datatype == list, 'datatype must be a list or dict(list)'

        # This is for PET to work properly (should be removed in future versions)
        self.input_dict = options
        self.true_order = [self.reporttype, options['reportpoint']]
        self.steps = [i for i in range(len(self.true_order[1]))]

        # Extract adjoint options
        #---------------------------------------------------------------------------------------------------------
        if 'adjoints' in options:
            self.compute_adjoints = True

            self.adjoint_info = {}
            for datatype in options['adjoints']:
                
                # Determine if rate or volume and phase. 
                if datatype in ['WOPT', 'WGPT', 'WWPT', 'WLPT']:
                    rate  = False
                    phase = {
                        'WOPT': 'oil',
                        'WGPT': 'gas',
                        'WWPT': 'water',
                        'WLPT': 'liquid',
                    }[datatype]
                
                elif datatype in ['WOPR', 'WGPR', 'WWPR', 'WLPR']:
                    rate  = True
                    phase = {
                        'WOPR': 'oil',
                        'WGPR': 'gas',
                        'WWPR': 'water',
                        'WLPR': 'liquid',
                    }[datatype]


                # Determine steps
                steps = options['adjoints'][datatype].get('steps', 'acc')
                accumulative= False

                if steps == 'acc':
                    steps = [self.steps[-1]]
                    accumulative = True
                elif steps == 'all':
                    steps = self.steps
                    accumulative = False
                elif isinstance(steps, int):
                    accumulative = False
                    steps = [steps]
                
                well_ids = options['adjoints'][datatype]['well_id']
                parameters = options['adjoints'][datatype]['parameters']

                # Ensure well_ids and parameters are lists
                well_ids = well_ids if isinstance(well_ids, (list, tuple)) else [well_ids]
                parameters = parameters if isinstance(parameters, (list, tuple)) else [parameters]

                for wid in well_ids:
                    self.adjoint_info[f'{datatype}:{wid}'] = {
                        'rate': rate,
                        'phase': phase,
                        'well_id': wid,
                        'parameters': parameters,
                        'steps': steps,
                        'accumulative': accumulative,
                    }
        #---------------------------------------------------------------------------------------------------------

    def __call__(self, inputs: dict):

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
                     

    def run_fwd_sim(self, input: dict, idn: int=0, delete_folder: bool=True):
        '''
        Run forward simulation for given input parameters.

        Parameters
        ----------
        input: dict
            Input parameters for the simulation.

        idn: int, optional
            Ensemble member ID, by default 0.

        delete_folder: bool, optional
            Whether to delete the simulation folder after running, by default True.

        Returns
        -------
            output: Union[dict, list, pd.DataFrame]
                Simulation output in the specified format.
        '''
        from juliacall import Main as jl
        from jutuldarcy import convert_to_pydict
        jl.seval("using JutulDarcy, Jutul")

        # Include ensemble member id in input dict
        input['member'] = idn
        
        # Make simulation folder
        folder = f'En_{idn}'
        os.makedirs(folder)

        # Render makofile
        self.render_makofile(self.makofile, folder, input)

        # Enter simulation folder and run simulation
        os.chdir(folder)

        # Setup case
        case = jl.setup_case_from_data_file(self.datafile)

        # Get some grid info
        nx = case.input_data["GRID"]["cartDims"][0]
        ny = case.input_data["GRID"]["cartDims"][1]  
        nz = case.input_data["GRID"]["cartDims"][2]
        grid = (nx, ny, nz)
        actnum = np.array(case.input_data["GRID"]["ACTNUM"]) # Shape (nx, ny, nz)
        actnum_vec = actnum.flatten(order='F')  # Fortran order flattening

        # Simulate and get results
        jlres = jl.simulate_reservoir(case, info_level=-1)
        pyres = convert_to_pydict(jlres, case, units='metric')
        pyres = self.results_to_dataframe(pyres, self.datatype, jlcase=case, jl_import=jl)

        # Convert output to requested format
        if not self.out_format == 'dataframe':
            if self.out_format == 'dict':
                output = pyres.to_dict(orient='list')
            elif self.out_format == 'list':
                output = []
                for idx, row in pyres.iterrows():
                    output.append(row.to_dict())
        else:
            output = pyres

        
        # Compute adjoints
        if self.compute_adjoints:

            # Initialize adjoint dataframe
            colnames = []
            for key in self.adjoint_info:
                for param in self.adjoint_info[key]['parameters']:
                    colnames.append((key, param))

            adjoints = pd.DataFrame(columns=pd.MultiIndex.from_tuples(colnames), index=self.true_order[1])
            adjoints.index.name = self.true_order[0]
            attrs = {}

            # Initialize progress bar
            pbar = tqdm(
                adjoints.keys(), 
                desc=f'Solving adjoints for En_{idn}',
                position=idn+1,
                leave=False,
                unit='obj',
                dynamic_ncols=False,
                **PBAR_OPTS
            )
            
            # Loop over adjoint objectives
            for col in adjoints.columns.levels[0]:
                info = self.adjoint_info[col]

                funcs = get_well_objective(
                    well_id=info['well_id'],
                    rate_id=info['phase'],
                    step_id=info['steps'],
                    rate=info['rate'],
                    accumulative=info['accumulative'],
                    jl_import=jl
                )

                # Define objective function
                funcs = funcs if isinstance(funcs, list) else [funcs]
                grads = []
                for func in funcs:
                    # Compute adjoint gradient
                    grad = jl.JutulDarcy.reservoir_sensitivities(
                        case, 
                        jlres, 
                        func,
                        include_parameters=True,
                    )
                    grads.append(grad)

                # Extract and store gradients in adjoint dataframe
                for g, grad in enumerate(grads):
                    for param in info['parameters']:
                        index = self.true_order[1][info['steps'][g]]
                        
                        if param.lower() == 'poro':
                            grad_param = np.array(grad[jl.Symbol("porosity")])
                            grad_param = _expand_to_active_grid(grad_param, actnum_vec, fill_value=0)
                            adjoints.at[index, (col, param)] = grad_param
                            attrs[(col, param)] = {'unit': 'Sm3'}

                        elif param.lower().startswith('perm'):
                            grad_param = np.array(grad[jl.Symbol("permeability")])
                            mD_per_m2 = _convert_from_si(1.0, 'darcy', jl) * 1e3
                            grad_param  = grad_param/mD_per_m2  # Convert from m2 to mD

                            if info['rate']:
                                days_per_sec = _convert_from_si(1.0, 'day', jl)
                                grad_param = grad_param/days_per_sec
                                attrs[(col, param)] = {'unit': 'Sm3/(day∙mD)'}
                            else:
                                attrs[(col, param)] = {'unit': 'Sm3/mD'}

                            if param.lower() == 'permx':
                                grad_param = grad_param[0]
                            elif param.lower() == 'permy':
                                grad_param = grad_param[1]
                            elif param.lower() == 'permz':
                                grad_param = grad_param[2]
                            
                            grad_param = _expand_to_active_grid(grad_param, actnum_vec, fill_value=0)
                            adjoints.at[index, (col, param)] = grad_param
                        else:
                            raise ValueError(f'Param: {param} not supported for adjoint sensitivity')
                
                
                # Update progress bar
                pbar.update(1)
            pbar.close()
            adjoints.attrs = attrs

        os.chdir('..')
        

        # Delete simulation folder
        if delete_folder:
            shutil.rmtree(folder)
        
        if self.compute_adjoints:
            return output, adjoints
        else:
            return output

    def render_makofile(self, makofile: str, folder: str, input: dict):
        '''
        Render makofile.mako to makofile.DATA using input
        '''
        self.datafile = makofile.replace('.mako', '.DATA')
        template = Template(filename=makofile)
        with open(os.path.join(folder, self.datafile), 'w') as f:
            f.write(template.render(**input))


    def results_to_dataframe(self, res: dict, datatypes: list, jlcase=None, jl_import=None) -> pd.DataFrame:
        
        df = pd.DataFrame(columns=datatypes, index=self.true_order[1])
        df.index.name = self.true_order[0]
        attrs = {}

        for key in datatypes:
            key_upper = key.upper()

            # Check if key is FIELD data
            if key_upper in res['FIELD']:
                df[key] = res['FIELD'][key_upper]
                attrs[key_upper] = {'unit': _metric_unit(key_upper)}

            # Check if key is WELLS data (format: "DATA:WELL" or "DATA WELL")
            elif ':' in key_upper or ' ' in key_upper:
                data_id, well_id = key_upper.replace(':', ' ').split(' ')
                df[key] = res['WELLS'][well_id][data_id]
                attrs[key_upper] = {'unit': _metric_unit(data_id)}

            elif key_upper in [str(k) for k in jlcase.input_data["GRID"].keys()] and (jlcase is not None):
                value = jlcase.input_data["GRID"][f"{key_upper}"]
                
                if key_upper.startswith('PERM'):
                    value = _convert_from_si(value, 'darcy', jl_import)
                    value = np.array(value) * 1e3 # Darcy to mD
                    attrs[key_upper] = {'unit': 'mD'}
                else:
                    attrs[key_upper] = {'unit': _metric_unit(key_upper)}

                try:
                    df.at[df.index[0], key] = np.array(value)
                except:
                    df.at[df.index[0], key] = value
                
            else:
                raise KeyError(f'Data type {key} not found in simulation results')
            
        df.attrs = attrs
        return df
    
    
def _symdict_to_pydict(symdict, jl_import):
    '''Convert a Julia symbolic dictionary to a Python dictionary recursively.'''
    pydict = {}
    for key, value in symdict.items():
        if jl_import.isa(value, jl_import.AbstractDict):
            pydict[str(key)] = _symdict_to_pydict(value, jl_import)
        else:
            pydict[str(key)] = value
    return pydict
    
def _expand_to_active_grid(param, active, fill_value=np.nan):

    if len(param) == active.sum():
        val = []
        i = 0
        for cell in active:
            if cell == 1:
                val.append(param[i])
                i += 1
            else:
                val.append(fill_value)
    elif len(param) == len(active):
        val = param
    else:
        raise ValueError('Parameter length does not match number of active cells')
    
    return np.array(val)


def _convert_from_si(value, unit, jl_import):
    '''Convert value from SI units to specified unit using JutulDarcy conversion.'''
    return jl_import.Jutul.convert_from_si(value, jl_import.Symbol(unit))

def _metric_unit(key: str) -> str:
    '''Return metric unit for given key.'''
    unit_map = {
        'PORO': '',
        'PERMX': 'mD',
        'PERMY': 'mD',
        'PERMZ': 'mD',
        #---------------------
        'FOPT': 'Sm3',
        'FGPT': 'Sm3',
        'FWPT': 'Sm3',
        'FWLT': 'Sm3',
        'FWIT': 'Sm3',
        #---------------------
        'FOPR': 'Sm3/day',
        'FGPR': 'Sm3/day',
        'FWPR': 'Sm3/day',
        'FLPR': 'Sm3/day',
        'FWIR': 'Sm3/day',
        #---------------------
        'WOPR': 'Sm3/day',
        'WGPR': 'Sm3/day',
        'WWPR': 'Sm3/day',
        'WLPR': 'Sm3/day',
        'WWIR': 'Sm3/day',
    }
    if key.upper() in unit_map:
        return unit_map[key.upper()]
    else:
        return 'Unknown'


def get_well_objective(well_id, rate_id, step_id, rate=True, accumulative=True, jl_import=None):
    '''
    Create a Julia objective function for well-based adjoint sensitivity analysis.

    This function generates JutulDarcy objective functions that compute well quantities
    of interest (QOI) for specific phases. The objective can target all timesteps,
    specific timesteps, or a single timestep, and can return either instantaneous
    rates or cumulative volumes.

    Parameters
    ----------
    well_id : str
        Identifier of the well for which to compute the objective.
    rate_id : str
        Phase type identifier. Supported values:
        - 'mass': Total surface mass rate
        - 'liquid': Surface liquid rate
        - 'water': Surface water rate
        - 'oil': Surface oil rate
        - 'gas': Surface gas rate
        - 'rate': Total volumetric rate
    step_id : int, list, np.ndarray, or None
        Timestep specification:
        - None: Compute objective for all timesteps (cumulative)
        - int: Compute objective for a single specific timestep
        - list/array: Compute objectives for multiple specific timesteps
    rate : bool, optional
        If True (default), returns instantaneous rate at timestep(s).
        If False, multiplies rate by dt for cumulative volume contribution.
    jl_import : module, optional
        Julia Main module from juliacall. If None, will import automatically.

    Returns
    -------
    function or list of functions
        - Single Julia objective function if step_id is None or int
        - List of Julia objective functions if step_id is a list/array

    Raises
    ------
    ValueError
        If rate_id is not one of the supported phase types.

    Examples
    --------
    >>> obj = get_well_objective('PROD1', 'oil', None, rate=False)
    >>> obj = get_well_objective('INJ1', 'water', 10, rate=True)
    >>> objs = get_well_objective('PROD2', 'gas', [5, 10, 15], rate=True)
    '''

    if jl_import is None:
        from juliacall import Main as jl_import
        jl_import.seval('using JutulDarcy')

    rate_id_map = {
        'mass': 'TotalSurfaceMassRate',
        'liquid': 'SurfaceLiquidRateTarget',
        'water': 'SurfaceWaterRateTarget',
        'oil': 'SurfaceOilRateTarget',
        'gas': 'SurfaceGasRateTarget',
        'rate': 'TotalRateTarget'
    }
    if rate_id not in rate_id_map:
        raise ValueError(f'Unknown rate type: {rate_id}')
    rate_id = rate_id_map[rate_id]

    if rate:
        dt = ''
    else:
        dt = 'dt*'

    # Case 1: Sum of all timesteps
    #-----------------------------------------------------------------------------
    if accumulative:
        jl_import.seval(f"""
        function objective_function(model, state, dt, step_i, forces)
            rate = JutulDarcy.compute_well_qoi(
                model, 
                state, 
                forces, 
                Symbol("{well_id}"), 
                {rate_id}
            )
            return {dt}rate
        end
        """)
        return jl_import.objective_function
    #-----------------------------------------------------------------------------
    
    # Case 2: Multiple timesteps
    #-----------------------------------------------------------------------------
    elif isinstance(step_id, (list, np.ndarray)):
        objective_steps = []
        for sid in step_id:
            jl_import.seval(f"""
            function objective_function_{sid}(model, state, dt, step_i, forces)
                if step_i != {sid+1}
                    return 0.0
                else
                    rate = JutulDarcy.compute_well_qoi(
                        model, 
                        state, 
                        forces, 
                        Symbol("{well_id}"), 
                        {rate_id}
                    )
                    return {dt}rate
                end
            end
            """)
            objective_steps.append(jl_import.seval(f'objective_function_{sid}'))
        return objective_steps
    #-----------------------------------------------------------------------------

    # Case 3: Single timestep
    #-----------------------------------------------------------------------------
    else:
        jl_import.seval(f"""
        function objective_function(model, state, dt, step_i, forces)
            if step_i != {step_id+1}
                return 0.0
            else
                rate = JutulDarcy.compute_well_qoi(
                    model, 
                    state, 
                    forces, 
                    Symbol("{well_id}"), 
                    {rate_id}
                )
                return {dt}rate
            end
        end
        """)
        return jl_import.objective_function
    #-----------------------------------------------------------------------------
