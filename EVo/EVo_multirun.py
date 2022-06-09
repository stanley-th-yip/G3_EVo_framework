import os
import yaml
import glob
import pandas as pd
import numpy as np
from itertools import product
from shutil import copyfile

# Creates/amends the input parameter file for evo with a specific run from inside the gridsearch
def amend_env(file, **kwargs):

    with open(file, 'r') as f:
        env_doc = yaml.full_load(f)

    for param, val in kwargs.items():
        if type(val) == np.float64:
            env_doc[param] = float(val)
        else:
            env_doc[param] = val

    with open('multirun.yaml', "w") as f:
        yaml.dump(env_doc, f)

# set up with arrays or lists of numbers you want to iterate through, will gridsearch the whole lot.
def multirun(**kwargs):
    
    # delete any pre-existing multirun setup files
    if os.path.exists('multirun.yaml'):
        os.remove('multirun.yaml')
    
    # Deletes any existing multirun output files to avoid confusion. 
    if os.path.exists('Output/output_0.csv'):
        filelist = glob.glob("Output/output*")
        for afile in filelist:
            os.remove(afile)

    #creates an iterable object where each 'column' is in the order given here.
    options = product(*kwargs.values())
    i=0
    keys=kwargs.keys()
    onerun={}    
    
    for run in options:
        for a, b in zip(keys, run):
            onerun[a] = b
        
        amend_env('env.yaml', **onerun)
        onerun = {}

        # Runs EVo
        os.system('python dgs.py chem.yaml multirun.yaml')

        # Copies the dgs_output file into a separate file ready to be run again.
        copyfile('Output/dgs_output.csv', f'Output/output_{i}.csv')


        i+=1

# ---------------------------------------------------------
# SET THE PARAMETERS WE WANT, THEN SEARCH OVER THESE RANGES
# ---------------------------------------------------------

def gridsearch_setsearch(set_attr={}, search={}):
    
    def my_product(inp):
        return [dict(zip(inp.keys(), values)) for values in product(*inp.values())]
    
    # delete any pre-existing multirun setup files
    if os.path.exists('multirun.yaml'):
        os.remove('multirun.yaml')

    # create a list of gridsearch runs for experiments I want data for
    run_options = my_product(set_attr)

    # create a list of set-up options to run through for each experiment
    val_options =  my_product(search)

    for run in run_options:  # this is the run we want fixed, then search over these values:
        
        name = [str(x) for x in run.values()]  # creates a filename for if the run is successfull
        name = '_'.join(name)
        
        for vals in val_options:
            # if we've already got a saved file matching 'run', break
            if os.path.exists(f'Output/output_{name}.csv'):
                break
            else:
                onerun = {**run, **vals}  # this is one run.

                amend_env('env.yaml', **onerun)

                # Runs EVo
                result = os.system('python dgs.py chem.yaml multirun.yaml')

                if result == 0:  # if the run doesn't fail for any reason
                    copyfile('Output/dgs_output.csv', f'Output/output_{name}.csv')

#multirun(FO2_buffer_START=[1, 0.5, 0, -0.5, -1, -1.5, -2], ATOMIC_C=[150, 550, 700])    #1, 0.5, 0, -0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4, -4.5, -5
gridsearch_setsearch(set_attr={'FO2_buffer_START':[-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5]}, search={'WTH2O_START':np.linspace(0.000112, 0.00004, 20), 'WTCO2_START':np.concatenate((np.linspace(4e-8, 1e-8, 7), np.linspace(9.5e-9, 6e-9, 8)))}) #np.concatenate((np.linspace(1.05e-8, 1e-8, 5), np.linspace(9.5e-9, 1.5e-9, 17)))





