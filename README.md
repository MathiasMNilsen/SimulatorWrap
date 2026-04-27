# Subsurface
Repository containing some standard simulation wrappers. For a detailed description of how to build the wrappers,
see the documentation for PET.

## Installation
Clone the repository and install it in editable mode by running
```bash
python3 -m pip install -e .
```

## Examples
The example demonstrates usage of [OPM-flow](https://opm-project.org/). To reproduce, ensure that OPM-flow is installed and in the path. The full noteboox is [here](https://github.com/Python-Ensemble-Toolbox/SimulatorWrap/tree/main/Example), we employ the [3Spot example](https://github.com/Python-Ensemble-Toolbox/Examples/tree/main/3Spot) as a template for this example. All files are Example folder.


```python
# import the flow class
from subsurface.multphaseflow.opm import flow
# import datatime
import datetime as dt
# import numpy
import numpy as np
```

To initialllize the wrapper such that we get the needed outputs we must specify multiple inputs. This will typically be extracted from a config .toml file.


```python
# a dictionary containing relevant information
input_dict = {'parallel':2,
             'simoptions': [['sim_flag', '--tolerance-mb=1e-5 --parsing-strictness=low']],
             'sim_limit': 4,
             'reportpoint': [dt.datetime(1994,2,9,00,00,00),
                            dt.datetime(1995,1,1,00,00,00),
                            dt.datetime(1996,1,1,00,00,00),
                            dt.datetime(1997,1,1,00,00,00),
                            dt.datetime(1998,1,1,00,00,00),
                            dt.datetime(1999,1,1,00,00,00)],
            'reporttype': "dates",
            'datatype': [
                "fopt",
                "fgpt",
                "fwpt",
                "fwit",
                "rft:pro-1"],
             'runfile':'3well'}

# name of the runfile
filename = '3WELL'
```


```python
# Generate an instance of the simulator class
sim = flow(input_dict=input_dict,filename=filename)
```


```python
# Setup simulator
sim.setup_fwd_run(redund_sim=None)
```

The point of the simulator wrapper in PET is to generate the simulation response for an ensemble of parameters. The mako template file needs to render a DATA file for each uncertain parameter. Hence, the syntax of the mako file need to match the test one wants to run. It is up to the user to specify this file. In the 3Spot case, the uncertain parameter consists of the bhp controll for the wells. In the wrapper this is a dictionary with keys matching the variablename in the mako file.


```python
state = {'injbhp':np.array([280,245]),
         'prodbhp':np.array([110])}
```


```python
# we can now run the flow simulator
pred = sim.run_fwd_sim(state,member_i=0,del_folder=True)
```


```python
pred
```


    [{
        'fopt': array(1.34704602),
        'fgpt': array(0.),
        'fwpt': array(0.00086854),
        'fwit': array(2.7143116),
        'rft:pro-1': array(nan)},
        {'fopt': array(277.52658081),
        'fgpt': array(0.),
        'fwpt': array(0.77613616),
        'fwit': array(480.96279907),
        'rft:pro-1': array([161.88829, 162.72838], dtype='>f4')},
        {'fopt': array(573.01849365),
        'fgpt': array(0.),
        'fwpt': array(1.46081805),
        'fwit': array(923.75933838),
        'rft:pro-1': array(nan)},
        {'fopt': array(866.71929932),
        'fgpt': array(0.),
        'fwpt': array(2.07612634),
        'fwit': array(1355.08740234),
        'rft:pro-1': array([160.21707, 161.05704], dtype='>f4')},
        {'fopt': array(1156.9161377),
        'fgpt': array(0.),
        'fwpt': array(2.6172893),
        'fwit': array(1787.93688965),
        'rft:pro-1': array(nan)},
        {'fopt': array(1445.58898926),
        'fgpt': array(0.),
        'fwpt': array(3.12104678),
        'fwit': array(2229.38110352),
        'rft:pro-1': array([159.58162, 160.4215 ], dtype='>f4')
        }]


```python
# we can make simple plot using matplotlib
import matplotlib.pyplot as plt
# Display plots inline in Jupyter
%matplotlib inline
```


```python
plt.figure(figsize=(10, 6));
plt.plot(input_dict['reportpoint'],np.concatenate(np.array([el['fopt'].flatten() for el in pred])));
plt.title('Field Oil Production Total');
plt.xlabel('Report dates');
plt.ylabel('STB');
```

<h1 align="center">
<img src="./Example/README_11_0.png">
</h1><br>


```python
plt.figure(figsize=(10, 6));
plt.plot(input_dict['reportpoint'],np.concatenate(np.array([el['fgpt'].flatten() for el in pred])));
plt.title('Field Gas Production Total');
plt.xlabel('Report dates');
plt.ylabel('STB');
```


<h1 align="center">
<img src="./Example/README_12_0.png">
</h1><br>



```python
plt.figure(figsize=(10, 6));
plt.plot(input_dict['reportpoint'],np.concatenate(np.array([el['fwpt'].flatten() for el in pred])));
plt.title('Field Water Production Total');
plt.xlabel('Report dates');
plt.ylabel('STB');
```


<h1 align="center">
<img src="./Example/README_13_0.png">
</h1><br>



```python
plt.figure(figsize=(10, 6));
plt.plot(input_dict['reportpoint'],np.concatenate(np.array([el['fwit'].flatten() for el in pred])));
plt.title('Field Water Injection Total');
plt.xlabel('Report dates');
plt.ylabel('STB');
```


<h1 align="center">
<img src="./Example/README_14_0.png">
</h1><br>


```python
import matplotlib.dates as mdates

plt.figure(figsize=(10, 6))
ax = plt.gca()
for reportpoint, rft_values in zip(input_dict['reportpoint'], [el['rft:pro-1'] for el in pred]):
    rft_values = np.asarray(rft_values).flatten()
    if rft_values.size == 1 and np.isnan(rft_values[0]):
        continue
    rft_values = rft_values[~np.isnan(rft_values)]
    if rft_values.size == 0:
        continue
    ax.plot([reportpoint] * rft_values.size, rft_values, color='C0')
ax.set_xlim(dt.datetime(input_dict['reportpoint'][0].year, 1, 1), max(input_dict['reportpoint']) + dt.timedelta(days=30))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_title('RFT PRO-1');ax.set_xlabel('Report dates');ax.set_ylabel('Bar');
```

<h1 align="center">
<img src="./Example/README_15_0.png">
</h1><br>
