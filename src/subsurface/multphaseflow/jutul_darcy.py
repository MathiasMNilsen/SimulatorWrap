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
#────────────────────────────────────────────────────


__author__ = 'Mathias Methlie Nilsen'
__all__ = ['JutulDarcyWrapper']


#────────────────────────────────────────────────────────────────────────────────────
os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes'
os.environ['PYTHON_JULIACALL_THREADS'] = 'auto'
os.environ['PYTHON_JULIACALL_OPTLEVEL'] = '3'
warnings.filterwarnings('ignore', message='.*juliacall module already imported.*')
#────────────────────────────────────────────────────────────────────────────────────


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
                - 'platform': 'Python' or 'Julia' (default: 'Python').

        References
        ----------
        [1] Møyner, O. (2025).
            JutulDarcy.jl – a fully differentiable high-performance reservoir simulator
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
        self.platform   = options.get('platform', 'Python')
        self.datafile   = None
        self.compute_adjoint = False

        # This is for PET to work properly (should be removed in future versions)
        self.input_dict = options
        self.true_order = [self.reporttype, options['reportpoint']]

        # Adjoint information
        if 'well_adjoint_info' in options:
            self.compute_adjoint = True
            self.adjoint_info = {'wells': {}}

            for well_id, well_info in options['well_adjoint_info'].items():
                well_obj = well_info['objective']
                well_var = well_info['variable']

                if isinstance(well_obj, str):
                    well_obj = [well_obj]
                if isinstance(well_var, str):
                    well_var = [well_var]

                for obj in well_obj:
                    if not obj in ['mass', 'liquid', 'water', 'oil', 'gas', 'rate']:
                        raise ValueError(f'Adjoint objective {obj} not supported')
                    
                self.adjoint_info['wells'][well_id] = {
                    'objective': well_obj,
                    'variable': well_var
                }
                

    def run_fwd_sim(self, input: dict, idn: int=0, delete_folder: bool=True) -> Union[dict|list|pd.DataFrame]:
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

        # Include ensemble member id in input dict
        input['member'] = idn
        
        # Make simulation folder
        folder = f'En_{idn}'
        os.makedirs(folder)

        # Render makofile
        self.render_makofile(self.makofile, folder, input)

        # Enter simulation folder and run simulation
        os.chdir(folder)

        if self.platform == 'Python':
            # Needs to be imported here for multiprocessing to work
            from jutuldarcy import simulate_data_file
            pyres = simulate_data_file(
                data_file_name=self.datafile, 
                convert=True, # Convert to output dictionary
                units='si',   # Use SI units (Sm3 and so on)
                info_level=-1 # No terminal output
            )
        elif self.platform == 'Julia':
            from juliacall import Main as jl
            from jutuldarcy import convert_to_pydict
            jl.seval("using JutulDarcy, Jutul")

            case  = jl.setup_case_from_data_file(self.datafile)
            jlres = jl.simulate_reservoir(case, info_level=-1)
            pyres = convert_to_pydict(jlres, case, units='si')

            # TODO: Make sure the gradient computation works (this example is hardcoded, for a specific case)
            if self.compute_adjoint:

                # TODO: There might be a better way of structuring the gradient info (this is very preliminary)!
                gradients = {}
                for well_id, well_info in self.adjoint_info['wells'].items():
                    gradients[well_id] = {}
                    for obj in well_info['objective']:
                        for var in well_info['variable']:

                            # Define objective function
                            obj_func = self.well_volume_objective(
                                well_id=well_id, 
                                phase=obj, 
                                jl_import=jl
                            )

                            # Compute gradients
                            obj_grad = jl.JutulDarcy.reservoir_sensitivities(
                                case, 
                                jlres, 
                                obj_func,
                                include_parameters=True,
                            )
                            print(self.symdict_to_pydict(obj_grad.data, jl).keys())

                            # Select relevant gradients
                            if var == 'poro':
                                poro_grad = np.array(obj_grad[jl.Symbol("porosity")])
                                gradients[well_id][f'{obj}_of_poro'] = poro_grad
                            
                            perm_grad = np.array(obj_grad[jl.Symbol("permeability")])
                            if var == 'permx':
                                gradients[well_id][f'{obj}_of_permx'] = perm_grad[0]
                            if var == 'permy':
                                gradients[well_id][f'{obj}_of_permy'] = perm_grad[1]
                            if var == 'permz':
                                gradients[well_id][f'{obj}_of_permz'] = perm_grad[2]
                
                print(gradients)
                
        os.chdir('..')

        # Delete simulation folder
        if delete_folder:
            shutil.rmtree(folder)

        # Extract requested datatypes
        output = self.extract_datatypes(pyres, out_format=self.out_format)
        
        if self.compute_adjoint:
            return output, gradients
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


    def extract_datatypes(self, res: dict, out_format='list') -> Union[dict|list|pd.DataFrame]:
        '''
        Extract requested datatypes from simulation results.

        Parameters
        ----------
        res : dict
            Simulation results dictionary.
        out_format : str, optional
            Output format ('list[dict]', 'dict', 'dataframe'), by default 'list'.
        
        Returns
        -------
            Union[dict, list, pd.DataFrame]
                Extracted data in the specified format.
        '''
        out = {}
        for orginal_key in self.datatype:
            key = orginal_key.upper()

            # Check if key is FIELD data
            if key in res['FIELD']:
                out[orginal_key] = res['FIELD'][key]
            # Check if key is WELLS data (format: "DATA:WELL" or "DATA WELL")
            elif ':' in key or ' ' in key:
                data_id, well_id = key.replace(':', ' ').split(' ')
                out[orginal_key] = res['WELLS'][well_id][data_id]
            else:
                raise KeyError(f'Data type {key} not found in simulation results')
        
        # Format output
        if out_format == 'list':
            # Make into a list of dicts where each dict is a time step (pred_data format)
            out_list = []
            for i in range(len(res['DAYS'])):
                time_step_data = {key: np.array([out[key][i]]) for key in out}
                out_list.append(time_step_data)
            return out_list
        
        elif out_format == 'dict':
            out['DAYS'] = res['DAYS']
            return out

        elif out_format == 'dataframe':
            df = pd.DataFrame(data=out, index=res['DAYS'])
            return df
        
    
    def well_volume_objective(self, well_id, phase, jl_import):
        '''
        Define a well volume objective function for sensitivity analysis.

        Parameters
        ----------
        well_id : str
            Identifier of the well.
        phase : str
            Phase type for the well.

        Returns
        -------
            jl.objective_function
                Julia objective function for the specified well and target.
        '''
        #jl_import.seval('using JutulDarcy')

        if phase == 'mass':
            rate = 'TotalSurfaceMassRate'
        elif phase == 'liquid':
            rate = 'SurfaceLiquidRateTarget'
        elif phase == 'water':
            rate = 'SurfaceWaterRateTarget'
        elif phase == 'oil':
            rate = 'SurfaceOilRateTarget'
        elif phase == 'gas':
            rate = 'SurfaceGasRateTarget'
        elif phase == 'rate':
            rate = 'TotalRateTarget'
        else:
            raise ValueError(f'Unknown phase type: {phase}')

        jl_import.seval(f"""
        function objective_function(model, state, dt, step_i, forces)
            rate = JutulDarcy.compute_well_qoi(model, state, forces, Symbol("{well_id}"), {rate})
            return dt*rate
        end
        """)

        return jl_import.objective_function
    

    def symdict_to_pydict(self, symdict, jl_import):
        '''Convert a Julia symbolic dictionary to a Python dictionary recursively.'''
        pydict = {}
        for key, value in symdict.items():
            if jl_import.isa(value, jl_import.AbstractDict):
                pydict[str(key)] = self.symdict_to_pydict(value, jl_import)
            else:
                pydict[str(key)] = value
        return pydict