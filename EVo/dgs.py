"""
dgs.py  PL: UPDATE THESE DESCRIPTIONS

--- SPECIES LIST ---

C    CO   CO_2 CH_4 [OCS]               | carbon, carbon monoxide, carbon dioxide, methane
H_2  H_2O H_2S      [CH4]               | hydrogen, water, hydrogen sulfide
SO_2 S_2            [H_2S OCS]          | sulfur dioxide, sulfur
O_2  OCS            [CO CO_2 H_2O SO_2 FeO Fe_2O_3]   | oxygen, carbonyl sulfide
FeO  Fe_2O_3                            | ferrous iron, ferric iron
N2                                      | nitrogen

--- P T RANGE ---

790 - 1400 deg C
300 MPa

-- MODULES ---

gasworks - calculate heterogeneous equilibria
solvgas - calculate homogeneous (gas phase) equilibria
ferric - calculate fo2 from ferric iron proportions

--- LITERATURE SOURCES ---

model description ---
Gaillard & Scaillet 2014, EPSL
Burgisser & Scaillet 2007
Moretti & Papale 2004
Holloway 1987
Burgisser 2015

solubility models ---
Iacono-marziano et al. 2012, EPSL
Gaillard & Scaillet 2009
Behrens and Gaillard 2006 (CO CH4 solubility)
Morizet et al. 2010 (CO CH4 solubility)
Hirschman 2012
Righter 2009


--- MODEL DESCRIPTION ---

Equilibrium constants and mass balance method
solve 1-11 simultaneously at each step

Add solver for graphite saturation (cco buffer)

--- FUTURE ADDITIONS ---

major gas species
Cl: HCl
F: HF

trace gas species
Na: NaCl
Cu: CuCl
Pb: PbS
B: H_2BO_3
"""

#------------------------------------------------------------------------
# IMPORTS
#------------------------------------------------------------------------

# python main [ ensure these are on your system ]
import os.path
import sys
import argparse
import time

# bundled scripts
from readin import *
from dgs_classes import *
from writeout import *
import solver
import solver_gasonly

#------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------

def main(f_chem, f_env, f_out):
    print("Reading in from:")
    print("Chemistry file:", f_chem)
    print("Environment file:", f_env)
    print("Output file:", f_out,"\n")

    # Instantiate the run, thermosystem, melt and output objects using the readin.py scripts
    run, sys, melt, out = readin(f_chem, f_env, f_out)  

    print("Set parameters:")
    run.print_conditions()

    # Setup the initial gas + melt chemistry, calculate the atomic masses of elements
    run, sys, melt, gas, molecules = set_init_chem(run, sys, melt)
    
    print("\nSystem chemistry (wt%)")
    sm = 0.   
    for k, v in cnvs.mol2wt(melt.cm_dry).items():
        print(k, ':', "{0:0.3Ng}".format(v*100))
        sm += v*100    
    print("Total =", round(sm, 2), "\n")

    # print output specifications if made
    if f_out:
        out.print_conditions()  

    if run.FO2_SET == True:
        print(f"Mass of system following buffering to log {sys.FO2:.3}, FMQ {cnvt.fo2_2fmq(log10(exp(sys.FO2)),sys.T,sys.P,run.FMQ_MODEL):+.3} = \n{sys.M} g (We no longer update this - assume the user specified system mass is what they want for the entire system, with proper feo/fe2o3 partitioning and volatiles added in, not just the melt)")

    else:
        print(f"System mass = {sys.M} g")

    # -------------------------------------------------------------------
    # CALCULATION LOOP --------------------------------------------------
    # -------------------------------------------------------------------

    if run.SINGLE_STEP == True:
        solver.decompress(run, sys, melt, gas, molecules)
        sys.P_track.append(float(sys.P))

        if run.OCS == True:
            sys.OCS = True
            sys.SC = ["H2O", "O2", "H2", "CO2", "CO", "CH4", "S2", "H2S", "SO2", "OCS"]
            sys.P = run.P_STOP
            solver_gasonly.gas_only(run, sys, melt, gas, molecules)

            print("The total gas weight percent is %.2f %% and the gas volume fraction is %.2f %% "
              "\nThe run has finished." % ((sys.WgT[-1] * 100), (sys.GvF[-1] * 100)))

            print("Pre-OCS, the gas is partitioned as (mol %%): \n O2: %.4e %% \n H2: %.4f %% \n H2O: %.4f %% \n CO2: %.4f %% \n CO: %.4f %% \n CH4: %.4e %% \n SO2: %.4f %% \n S2: %.4f %% \n H2S: %.4f %% \n" % (
            gas.mO2[-2]*100, gas.mH2[-2]*100, gas.mH2O[-2]*100, gas.mCO2[-2]*100, gas.mCO[-2]*100, gas.mCH4[-2]*100, gas.mSO2[-2]*100, gas.mS2[-2]*100, gas.mH2S[-2]*100))

            print("Post-OCS, the gas is partitioned as (mol %%): \n O2: %.4e %% \n H2: %.4f %% \n H2O: %.4f %% \n CO2: %.4f %% \n CO: %.4f %% \n CH4: %.4e %% \n SO2: %.4f %% \n S2: %.4f %% \n H2S: %.4f %% \n OCS: %.4f %% \n" % (
            gas.mO2[-1]*100, gas.mH2[-1]*100, gas.mH2O[-1]*100, gas.mCO2[-1]*100, gas.mCO[-1]*100, gas.mCH4[-1]*100, gas.mSO2[-1]*100, gas.mS2[-1]*100, gas.mH2S[-1]*100, gas.mOCS[-1]*100))
            
            writeout_ocs(sys, gas, melt, sys.P_track) 
        else: 
            print("The total gas weight percent is %.2f %% and the gas volume fraction is %.2f %% "
                % ((sys.WgT[-1] * 100), (sys.GvF[-1] * 100)))
            print("The gas phase at %.0f bar is partitioned according to molar fraction as: \n O2: %s \n H2: %s \n H2O: %s \n CO2: %s \n CO: %s \n CH4: %s \n S2: %s \n SO2: %s \n H2S: %s \n N2: %s \n"
                % (sys.P, gas.mO2, gas.mH2, gas.mH2O, gas.mCO2, gas.mCO, gas.mCH4, gas.mS2, gas.mSO2, gas.mH2S, gas.mN2))
            writeout_file(sys, gas, melt, sys.P_track)

    elif run.SINGLE_STEP == False:
        while sys.P >= run.P_STOP:
            sys.P_track.append(float(sys.P))
            solver.decompress(run, sys, melt, gas, molecules)  # does one pressure step
            sys.pressure_step()
            
        if run.OCS == True:
            
            sys.OCS = True
            sys.SC = ["H2O", "O2", "H2", "CO2", "CO", "CH4", "S2", "H2S", "SO2", "OCS"]
            sys.P = run.P_STOP
            solver_gasonly.gas_only(run, sys, melt, gas, molecules)

            print("The pressure is %.0f bar, the total gas weight percent is %.2f %% and the gas volume fraction is %.2f %% "
              "\nThe run has finished." % (run.P_STOP, (sys.WgT[-1] * 100), (sys.GvF[-1] * 100)))

            print("Pre-OCS, the gas is partitioned as (mol %%): \n O2: %.4e %% \n H2: %.4f %% \n H2O: %.4f %% \n CO2: %.4f %% \n CO: %.4f %% \n CH4: %.4e %% \n SO2: %.4f %% \n S2: %.4f %% \n H2S: %.4f %% \n" % (
            gas.mO2[-2]*100, gas.mH2[-2]*100, gas.mH2O[-2]*100, gas.mCO2[-2]*100, gas.mCO[-2]*100, gas.mCH4[-2]*100, gas.mSO2[-2]*100, gas.mS2[-2]*100, gas.mH2S[-2]*100))

            print("Post-OCS, the gas is partitioned as (mol %%): \n O2: %.4e %% \n H2: %.4f %% \n H2O: %.4f %% \n CO2: %.4f %% \n CO: %.4f %% \n CH4: %.4e %% \n SO2: %.4f %% \n S2: %.4f %% \n H2S: %.4f %% \n OCS: %.4f %% \n" % (
            gas.mO2[-1]*100, gas.mH2[-1]*100, gas.mH2O[-1]*100, gas.mCO2[-1]*100, gas.mCO[-1]*100, gas.mCH4[-1]*100, gas.mSO2[-1]*100, gas.mS2[-1]*100, gas.mH2S[-1]*100, gas.mOCS[-1]*100))
            
            end = time.time()
            print("Run time is ", end-start)

            writeout_ocs(sys, gas, melt, sys.P_track)

        else:
        
            if (run.FIND_SATURATION == True or run.ATOMIC_MASS_SET == True) and sys.sat_conditions[0] < run.P_STOP:
                exit(f'Error: The saturation pressure ({sys.sat_conditions[0]:.3} bar) is lower than P_STOP ({run.P_STOP:.3} bar).\nPlease either lower P_STOP or change the melt initial fo2/melt volatile content.\nExiting...')
            
            print("The pressure is %.0f bar, the total gas weight percent is %.2f %% and the gas volume fraction is %.2f %% "
                "\nThe run has finished." % (run.P_STOP, (sys.WgT[-1] * 100), (sys.GvF[-1] * 100)))

            print("The gas is partitioned as (mol %%): \n O2: %.4e %% \n H2: %.4f %% \n H2O: %.4f %% \n CO2: %.4f %% \n CO: %.4f %% \n CH4: %.4e %% \n SO2: %.4f %% \n S2: %.4f %% \n H2S: %.4f %% \n" % (
            gas.mO2[-1]*100, gas.mH2[-1]*100, gas.mH2O[-1]*100, gas.mCO2[-1]*100, gas.mCO[-1]*100, gas.mCH4[-1]*100, gas.mSO2[-1]*100, gas.mS2[-1]*100, gas.mH2S[-1]*100))

            print("The first, middle, last gas molar fractions are: \n O2: %s   %s   %s \n H2: %s   %s   %s \n H2O: %s   %s   %s \n CO2: %s   %s   %s \n CO: %s   %s   %s \n CH4: %s   %s   %s \n SO2: %s   %s   %s \n S2: %s   %s   %s \n H2S: %s   %s   %s \n" % (
            gas.mO2[0],gas.mO2[(len(gas.mO2)//2)],gas.mO2[-1], gas.mH2[0],gas.mH2[(len(gas.mO2)//2)],gas.mH2[-1], gas.mH2O[0],gas.mH2O[(len(gas.mH2O)//2)],gas.mH2O[-1],
            gas.mCO2[0],gas.mCO2[(len(gas.mO2)//2)],gas.mCO2[-1], gas.mCO[0],gas.mCO[(len(gas.mO2)//2)],gas.mCO[-1], gas.mCH4[0],gas.mCH4[(len(gas.mH2O)//2)],gas.mCH4[-1],
            gas.mSO2[0],gas.mSO2[(len(gas.mO2)//2)],gas.mSO2[-1], gas.mS2[0],gas.mS2[(len(gas.mO2)//2)],gas.mS2[-1], gas.mH2S[0],gas.mH2S[(len(gas.mH2O)//2)],gas.mH2S[-1]))
            
            end = time.time()
            print("Run time is ", end-start)

            writeout_file(sys, gas, melt, sys.P_track)
            writeout_figs(sys, melt, gas, out, sys.P_track)

if __name__ == "__main__":
       
    # -------------------------------------------------------------------
    # SYSTEM SETUP ------------------------------------------------------
    # -------------------------------------------------------------------
    
    start = time.time()
    
    # Create the parser
    my_parser = argparse.ArgumentParser(prog='dgs', description='Run EVo: a thermodynamic magma degassing model')

    # Add the arguments
    my_parser.add_argument('chem',
                        metavar='chem.yaml',
                        help='the magma chemistry file')
    
    my_parser.add_argument('env',
                        metavar='env.yaml',
                        help='the run environment settings file')
    
    my_parser.add_argument('--output',
                        help='use selected output options from output.yaml file rather than default outputs')

    # Parse in files
    args = my_parser.parse_args()

    f_chem = args.chem          # set chemical compositions file
    f_env = args.env            # set environment file
    
    if args.output:
        f_out = args.output     # set output file as an optional input (i.e. if want to specify types of output rather than printing/ graphing all of them
        main(f_chem, f_env, f_out)
    else:
        f_out = None
        main(f_chem, f_env, f_out)

