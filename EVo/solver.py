# solver.py
"""STORES THE SOLVING EQUATIONS AND SOLVE ROUTINE"""

import constants as cnst
import conversions as cnvs
import gmpy2 as gp
import solvgas as sg
from scipy import optimize
import numpy as np
import warnings
import messages as msgs
import solubility_laws as sl


def decompress(run, sys, melt, gas, system):

    # --------------------------------------------------------------------------------------------------------
    # FUNCTION DEFINITIONS
    # --------------------------------------------------------------------------------------------------------

    # OH equations
    def oh_eqs(guesses=[1], fe=True):

        # x = mO2
        
        x = guesses[0]
        
        mH2O = (1 - x) / (1 + (H2O.Y / (sys.K["K1"] * H2.Y * (x * O2.Y * sys.P) ** 0.5)))

        mH2 = 1 - mH2O - x

        H = ((sys.atomicM['h'] / (2* cnst.m['h'])) - sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) - sl.h2_melt(mH2, H2, sys.P, name=run.H2_MODEL)) / (mH2O + mH2)
        
        if fe == True:
            ofe = sys.fe_equil(melt, x, O2)/cnst.m['o']
            atomico = sys.atomicM['o_tot']
        elif fe == False:
            ofe = 0
            atomico = sys.atomicM['o']
        
        return [((H * (mH2O + 2*x)) + sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) + ofe) - atomico/cnst.m['o']]


    # COH system ---------------------------------------------------------------------------------------------
    def coh_eqs(guesses=[], fe=True, graph=False):
        # x = mO2  y = mCO
        
        # x, y = guesses[0], guesses[1]

        # mCO2 = sys.K['K2'] * CO.Y*y*(O2.Y*x*sys.P)**0.5 / CO2.Y

        for g in guesses:
            if np.isnan(g):
                raise RuntimeWarning('NaN in guesses')

        if graph == False:
            x, y = guesses[0], guesses[1]

            mCO2 = (sys.K['K2'] * CO.Y * y * (O2.Y * x * sys.P) ** 0.5) / CO2.Y

        elif graph == True:
            x = guesses[0]

            mCO2 = sl.graphite_fco2(sys.T, sys.P, (O2.Y*x*sys.P))/(CO2.Y*sys.P)

            y = (CO2.Y * mCO2)/(sys.K['K2']*CO.Y*(O2.Y*x*sys.P)**0.5)   # mCO, dependent on CO2 from graphite saturation.

        mH2O = (-(1 + ((H2O.Y*sys.P)/(sys.K['K1']*H2.Y*sys.P*(O2.Y*x*sys.P)**0.5))) + ((1 + ((H2O.Y*sys.P)/(sys.K['K1']*H2.Y*sys.P*(O2.Y*x*sys.P)**0.5)))**2 - 4*CO2.Y*mCO2*(H2O.Y * sys.P)**2 / (sys.K['K3']*CH4.Y*(x*O2.Y*sys.P)**2)*-(1 - x - y - mCO2))**0.5)/(2*CO2.Y*mCO2*(H2O.Y * sys.P)**2 / (sys.K['K3']*CH4.Y*(x*O2.Y*sys.P)**2))

        mH2 = H2O.Y*mH2O/(sys.K['K1']*H2.Y*(O2.Y*x*sys.P)**0.5)

        mCH4 = (CO2.Y*mCO2*(H2O.Y*mH2O)**2) / (sys.K['K3']*CH4.Y*(O2.Y*x)**2)

        #C = ((sys.atomicM['c'] / cnst.m['c']) - sl.co2_melt((CO2.Y*mCO2*sys.P), CO2, (O2.Y*x*sys.P), sys.T, sys.P, melt, name=run.C_MODEL) - sl.co_melt((CO.Y*y*sys.P), sys.P, name=run.CO_MODEL) - sl.ch4_melt((CH4.Y*mCH4*sys.P), sys.P, name=run.CH4_MODEL)) / (y + mCO2 + mCH4)
        H = ((sys.atomicM['h'] / (2*cnst.m['h'])) - sl.h2_melt(mH2, H2, sys.P, name=run.H2_MODEL) - sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) - 2*sl.ch4_melt((CH4.Y*mCH4*sys.P), sys.P, name = run.CH4_MODEL)) / (mH2O + mH2 + 2*mCH4)

        if fe == True:
            ofe = sys.fe_equil(melt, x, O2)/cnst.m['o']
            atomico = sys.atomicM['o_tot']
        elif fe == False:
            ofe = 0
            atomico = sys.atomicM['o']

        # return [((C * (mH2O + 2*x + y + 2*mCO2)) + sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) + 2*sl.co2_melt((CO2.Y*mCO2*sys.P), CO2, (O2.Y*x*sys.P), sys.T, sys.P, melt, name=run.C_MODEL) + sl.co_melt((CO.Y*y*sys.P), sys.P, name = run.CO_MODEL) + ofe) - atomico/cnst.m['o'],
        
        # (C * (mH2O + mH2 + 2*mCH4) + sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) + sl.h2_melt(mH2, H2, sys.P, name=run.H2_MODEL) + 2*sl.ch4_melt((CH4.Y*mCH4*sys.P), sys.P, name = run.CH4_MODEL)) - sys.atomicM['h']/(2*cnst.m['h'])]
    
        if graph == True:
            melt.graph_current = cnvs.get_graphite(sys, melt, sys.P, CO2, y, mCO2, mCH4, x, CO2.Y, CO.Y, CH4.Y, O2.Y, H)/cnst.m['c']
        
            return [(H*(mH2O + 2*x + y + 2*mCO2) + sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) + 2*sl.co2_melt((CO2.Y*mCO2*sys.P), CO2, (O2.Y*x*sys.P), sys.T, sys.P, melt, name=run.C_MODEL) + sl.co_melt((CO.Y*y*sys.P), sys.P, name = run.CO_MODEL) + ofe) - (atomico/cnst.m['o'])]
                
        elif graph == False:
            melt.graph_current = gp.mpfr('0')
        
            return [(H*(mH2O + 2*x + y + 2*mCO2) + sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) + 2*sl.co2_melt((CO2.Y*mCO2*sys.P), CO2, (O2.Y*x*sys.P), sys.T, sys.P, melt, name=run.C_MODEL) + sl.co_melt((CO.Y*y*sys.P), sys.P, name = run.CO_MODEL) + ofe) - (atomico/cnst.m['o']),
            
            (H * (y + mCO2 + mCH4) + sl.co2_melt((CO2.Y*mCO2*sys.P), CO2, (O2.Y*x*sys.P), sys.T, sys.P, melt, name=run.C_MODEL) + sl.co_melt((CO.Y*y*sys.P), sys.P, name=run.CO_MODEL) + sl.ch4_melt((CH4.Y*mCH4*sys.P), sys.P, name = run.CH4_MODEL)) - (sys.atomicM['c']/(cnst.m['c']))]
        
    def soh_eqs(guesses=[1, 1], fe=True):
        # x = mO2, y = mS2

        x, y = guesses[0], guesses[1]
        
        mSO2 = (sys.K['K5'] * O2.Y * x * gp.sqrt(S2.Y * y * sys.P)) / SO2.Y

        mH2S = (1 - x - y - mSO2) / (
                1 + (H2S.Y / (sys.K['K1'] * sys.K['K4'] * H2.Y * gp.sqrt(S2.Y * y * sys.P))) + ((H2S.Y * gp.sqrt(O2.Y * x)) / (H2O.Y * sys.K['K4'] * gp.sqrt(S2.Y * y))))

        mH2 = H2S.Y * mH2S / (sys.K['K1'] * sys.K['K4'] * H2.Y * (S2.Y * y * sys.P) ** 0.5)

        mH2O = (H2S.Y * mH2S * gp.sqrt(O2.Y * x)) / (H2O.Y * sys.K['K4'] * gp.sqrt(S2.Y * y))

        mH2 = H2O.Y * mH2O / (H2.Y * sys.K['K1'] * gp.sqrt(O2.Y * x * sys.P))

        S = ((sys.atomicM['s'] / cnst.m['s']) - sl.sulfide_melt((S2.Y*y*sys.P), (O2.Y*x*sys.P), sys.P, sys.T, melt, name=run.SULFIDE_CAPACITY) - sl.sulfate_melt((S2.Y*y*sys.P), (O2.Y*x*sys.P), sys.P, sys.T, melt, run, name=run.SULFATE_CAPACITY)) / (2 * y + mH2S + mSO2)

        if fe == True:
            ofe = sys.fe_equil(melt, x, O2)/cnst.m['o']
            atomico = sys.atomicM['o_tot']
        elif fe == False:
            ofe = 0
            atomico = sys.atomicM['o']
        
        return [(S * (mH2O + 2 * x + 2 * mSO2) + sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) + ofe) - (atomico / cnst.m['o']),
        
        (S * (mH2 + mH2O + mH2S) + sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) + sl.h2_melt(mH2, H2, sys.P, name=run.H2_MODEL)) - (sys.atomicM['h'] / (2 * cnst.m['h']))]
    
   
    def cohs_eqs(guesses=[], fe=True, graph=False):
        # x = mCO, y = mS2, z = mO2

        # x, y, z = guesses[0], guesses[1], guesses[2]

        for g in guesses:
            if np.isnan(g):
                raise RuntimeWarning('NaN in guesses')

        if graph == False:
            x, y, z = guesses[0], guesses[1], guesses[2]

            mCO2 = (sys.K['K2'] * CO.Y * x * (O2.Y * z * sys.P) ** 0.5) / CO2.Y

        elif graph == True:
            y, z = guesses[0], guesses[1]

            mCO2 = sl.graphite_fco2(sys.T, sys.P, (O2.Y*z*sys.P))/(CO2.Y*sys.P)

            x = (CO2.Y * mCO2)/(sys.K['K2']*CO.Y*(O2.Y*z*sys.P)**0.5)   # mCO, dependent on CO2 from graphite saturation.
        
        # mCO2 = (sys.K['K2'] * CO.Y * x * (O2.Y * z * sys.P) ** 0.5) / CO2.Y

        mSO2 = (sys.K['K5'] * O2.Y * z * (S2.Y * y * sys.P) ** 0.5) / SO2.Y

        a = (CO2.Y*mCO2*H2O.Y**2)/(sys.K['K3']*CH4.Y*(O2.Y*z)**2)
        b = (H2O.Y/(sys.K['K1']*H2.Y*(O2.Y*z*sys.P)**0.5)) + (sys.K['K4']*H2O.Y*(S2.Y*y)**0.5/(H2S.Y*(O2.Y*z)**0.5)) + 1
        c = -(1-z-mCO2-x-mSO2-y)

        mH2O = (-b + (b**2-(4*a*c))**0.5)/(2*a)

        mH2 = (H2O.Y * mH2O) / (sys.K['K1'] * H2.Y * (O2.Y * z * sys.P) ** 0.5)

        mCH4 = (CO2.Y * mCO2 * (H2O.Y * mH2O) ** 2) / (sys.K['K3'] * CH4.Y * (O2.Y * z) ** 2)

        mH2S = (sys.K['K4']*H2O.Y*mH2O*(S2.Y*y)**0.5)/(H2S.Y*(O2.Y*z)**0.5)

        #C = ((sys.atomicM['c'] / cnst.m['c']) - sl.co2_melt((CO2.Y*mCO2*sys.P), CO2, (O2.Y*z*sys.P), sys.T, sys.P, melt, name=run.C_MODEL) - sl.co_melt((CO.Y*x*sys.P), sys.P, name=run.CO_MODEL) - sl.ch4_melt((CH4.Y*mCH4*sys.P), sys.P, name=run.CH4_MODEL)) / (x + mCO2 + mCH4)
        H = ((sys.atomicM['h'] / (2*cnst.m['h'])) - sl.h2_melt(mH2, H2, sys.P, name=run.H2_MODEL) - sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) - 2*sl.ch4_melt((CH4.Y*mCH4*sys.P), sys.P, name = run.CH4_MODEL)) / (mH2O + mH2 + mH2S + 2*mCH4)

        if fe == True:
            ofe = sys.fe_equil(melt, z, O2)/cnst.m['o']
            atomico = sys.atomicM['o_tot']
        elif fe == False:
            ofe = 0
            atomico = sys.atomicM['o']
        
        # return [(C*(2*z + 2*mSO2 + mH2O + x + 2*mCO2) + sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) + 2*sl.co2_melt((CO2.Y*mCO2*sys.P), CO2, (O2.Y*z*sys.P), sys.T, sys.P, melt, name=run.C_MODEL) + sl.co_melt((CO.Y*x*sys.P), sys.P, name = run.CO_MODEL) + ofe) - (atomico/cnst.m['o']),
        
        # (C * (mH2O + mH2 + mH2S + 2*mCH4) + sl.h2_melt(mH2, H2, sys.P, name=run.H2_MODEL) + sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) + 2*sl.ch4_melt((CH4.Y*mCH4*sys.P), sys.P, name = run.CH4_MODEL)) - (sys.atomicM['h']/(2*cnst.m['h'])),
        
        # (C*(mSO2 + mH2S + 2*y) + sl.sulfide_melt((S2.Y*y*sys.P), (O2.Y*z*sys.P), sys.P, sys.T, melt, name=run.SULFIDE_CAPACITY) + sl.sulfate_melt((S2.Y*y*sys.P), (O2.Y*z*sys.P), sys.P, sys.T, melt, run, name=run.SULFATE_CAPACITY)) - (sys.atomicM['s']/cnst.m['s'])]

        if graph == True:
            melt.graph_current = cnvs.get_graphite(sys, melt, sys.P, CO2, x, mCO2, mCH4, z, CO2.Y, CO.Y, CH4.Y, O2.Y, H)/cnst.m['c']
        
            return [(H*(2*z + 2*mSO2 + mH2O + x + 2*mCO2) + sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) + 2*sl.co2_melt((CO2.Y*mCO2*sys.P), CO2, (O2.Y*z*sys.P), sys.T, sys.P, melt, name=run.C_MODEL) + sl.co_melt((CO.Y*x*sys.P), sys.P, name = run.CO_MODEL) + ofe) - (atomico/cnst.m['o']),
                
            (H*(mSO2 + mH2S + 2*y) + sl.sulfide_melt((S2.Y*y*sys.P), (O2.Y*z*sys.P), sys.P, sys.T, melt, name=run.SULFIDE_CAPACITY) + sl.sulfate_melt((S2.Y*y*sys.P), (O2.Y*z*sys.P), sys.P, sys.T, melt, run, name=run.SULFATE_CAPACITY)) - (sys.atomicM['s']/cnst.m['s'])]

        elif graph == False:
            melt.graph_current = gp.mpfr('0')
        
            return [(H*(2*z + 2*mSO2 + mH2O + x + 2*mCO2) + sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) + 2*sl.co2_melt((CO2.Y*mCO2*sys.P), CO2, (O2.Y*z*sys.P), sys.T, sys.P, melt, name=run.C_MODEL) + sl.co_melt((CO.Y*x*sys.P), sys.P, name = run.CO_MODEL) + ofe) - (atomico/cnst.m['o']),
            
            (H * (x + mCO2 + mCH4) + sl.co2_melt((CO2.Y*mCO2*sys.P), CO2, (O2.Y*z*sys.P), sys.T, sys.P, melt, name=run.C_MODEL) + sl.co_melt((CO.Y*x*sys.P), sys.P, name=run.CO_MODEL) + sl.ch4_melt((CH4.Y*mCH4*sys.P), sys.P, name = run.CH4_MODEL)) - (sys.atomicM['c']/(cnst.m['c'])),
            
            (H*(mSO2 + mH2S + 2*y) + sl.sulfide_melt((S2.Y*y*sys.P), (O2.Y*z*sys.P), sys.P, sys.T, melt, name=run.SULFIDE_CAPACITY) + sl.sulfate_melt((S2.Y*y*sys.P), (O2.Y*z*sys.P), sys.P, sys.T, melt, run, name=run.SULFATE_CAPACITY)) - (sys.atomicM['s']/cnst.m['s'])]


    def cohsn_eqs(guesses=[], fe=True, graph=False):
        # w = mN2 x = mCO, y = mS2, z = mO2

        for g in guesses:
            if np.isnan(g):
                raise RuntimeWarning('NaN in guesses')

        if graph == False:
            w, x, y, z = guesses[0], guesses[1], guesses[2], guesses[3]

            mCO2 = (sys.K['K2'] * CO.Y * x * (O2.Y * z * sys.P) ** 0.5) / CO2.Y

        elif graph == True:
            w, y, z = guesses[0], guesses[1], guesses[2]

            mCO2 = sl.graphite_fco2(sys.T, sys.P, (O2.Y*z*sys.P))/(CO2.Y*sys.P)

            x = (CO2.Y * mCO2)/(sys.K['K2']*CO.Y*(O2.Y*z*sys.P)**0.5)   # mCO, dependent on CO2 from graphite saturation.

        mSO2 = (sys.K['K5'] * O2.Y * z * (S2.Y * y * sys.P) ** 0.5) / SO2.Y

        a = (CO2.Y*mCO2*H2O.Y**2)/(sys.K['K3']*CH4.Y*(O2.Y*z)**2)
        b = (H2O.Y/(sys.K['K1']*H2.Y*(O2.Y*z*sys.P)**0.5)) + (sys.K['K4']*H2O.Y*(S2.Y*y)**0.5/(H2S.Y*(O2.Y*z)**0.5)) + 1
        c = -(1-z-mCO2-x-mSO2-y-w)

        mH2O = (-b + (b**2-(4*a*c))**0.5)/(2*a)

        mH2 = (H2O.Y * mH2O) / (sys.K['K1'] * H2.Y * (O2.Y * z * sys.P) ** 0.5)

        mCH4 = (CO2.Y * mCO2 * (H2O.Y * mH2O) ** 2) / (sys.K['K3'] * CH4.Y * (O2.Y * z) ** 2)

        mH2S = (sys.K['K4'] * H2O.Y * mH2O * (S2.Y * y) ** 0.5) / (H2S.Y * (O2.Y * z) ** 0.5)

        N = ((sys.atomicM['n'] / cnst.m['n']) - sl.n_melt(w, (O2.Y*z*sys.P), sys.P, name=run.N_MODEL)) / (2*w)
            
        if fe == True:
            ofe = sys.fe_equil(melt, z, O2)/cnst.m['o']
            atomico = sys.atomicM['o_tot']
        elif fe == False:
            ofe = 0
            atomico = sys.atomicM['o']

        if graph == True:
            melt.graph_current = cnvs.get_graphite(sys, melt, sys.P, CO2, x, mCO2, mCH4, z, CO2.Y, CO.Y, CH4.Y, O2.Y, N)/cnst.m['c']

            return [(N*(2*z + 2*mSO2 + mH2O + x + 2*mCO2) + sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) + 2*sl.co2_melt((CO2.Y*mCO2*sys.P), CO2, (O2.Y*z*sys.P), sys.T, sys.P, melt, name=run.C_MODEL) + sl.co_melt((CO.Y*x*sys.P), sys.P, name = run.CO_MODEL) + ofe) - (atomico/cnst.m['o']),

            (N * (mH2O + mH2 + mH2S + 2*mCH4) + sl.h2_melt(mH2, H2, sys.P, name=run.H2_MODEL) + sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) + 2*sl.ch4_melt((CH4.Y*mCH4*sys.P), sys.P, name = run.CH4_MODEL)) - (sys.atomicM['h']/(2*cnst.m['h'])),

            (N*(mSO2 + mH2S + 2*y) + sl.sulfide_melt((S2.Y*y*sys.P), (O2.Y*z*sys.P), sys.P, sys.T, melt, name=run.SULFIDE_CAPACITY) + sl.sulfate_melt((S2.Y*y*sys.P), (O2.Y*z*sys.P), sys.P, sys.T, melt, run, name=run.SULFATE_CAPACITY)) - (sys.atomicM['s']/cnst.m['s'])]


        elif graph == False:
            melt.graph_current = gp.mpfr('0')

            return [(N*(2*z + 2*mSO2 + mH2O + x + 2*mCO2) + sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) + 2*sl.co2_melt((CO2.Y*mCO2*sys.P), CO2, (O2.Y*z*sys.P), sys.T, sys.P, melt, name=run.C_MODEL) + sl.co_melt((CO.Y*x*sys.P), sys.P, name = run.CO_MODEL) + ofe) - (atomico/cnst.m['o']),

            (N * (mH2O + mH2 + mH2S + 2*mCH4) + sl.h2_melt(mH2, H2, sys.P, name=run.H2_MODEL) + sl.h2o_melt(mH2O, H2O, sys.P, name=run.H2O_MODEL) + 2*sl.ch4_melt((CH4.Y*mCH4*sys.P), sys.P, name = run.CH4_MODEL)) - (sys.atomicM['h']/(2*cnst.m['h'])),

            (N*(mSO2 + mH2S + 2*y) + sl.sulfide_melt((S2.Y*y*sys.P), (O2.Y*z*sys.P), sys.P, sys.T, melt, name=run.SULFIDE_CAPACITY) + sl.sulfate_melt((S2.Y*y*sys.P), (O2.Y*z*sys.P), sys.P, sys.T, melt, run, name=run.SULFATE_CAPACITY)) - (sys.atomicM['s']/cnst.m['s']),
          
            (N * (x + mCO2 + mCH4) + sl.co2_melt((CO2.Y*mCO2*sys.P), CO2, (O2.Y*z*sys.P), sys.T, sys.P, melt, name=run.C_MODEL) + sl.co_melt((CO.Y*x*sys.P), sys.P, name = run.CO_MODEL) + sl.ch4_melt((CH4.Y*mCH4*sys.P), sys.P, name = run.CH4_MODEL)) - (sys.atomicM['c']/cnst.m['c'])]


    # Jacobian Newton-Rapheson -------------------------------------------------------------------------------
    def jac_newton(run, gas, mols):
        """
        Solves for the key variables in the next pressure step using newton rapheson.

        Takes the result from the previous pressure step as an initial guess,
        and runs scipy fsolve to optimize for the result. If the system does not
        converge, the pressure step is decreased by a factor of 10 and the system re-runs
        the calculation at a higher pressure.

        Args:
            run (class): active instance of RunDef class
            sys (class): active instance of ThermosystemDef class
            mols (list): all instances of the Molecule class, listed as a string.

        Returns:
            Appropriate number of variable solutions at the current pressure step.
        """

        # get initial guesses based on which volatile system is being run
        
        if run.GAS_SYS == "OH" or run.GAS_SYS == 'COH' or run.GAS_SYS == 'SOH':
            guessx = float(gas.mO2[-1])
            
            if run.GAS_SYS == 'COH':
                guessy = float(gas.mCO[-1])
            if run.GAS_SYS == 'SOH':
                guessy = float(gas.mS2[-1])
        
        elif run.GAS_SYS == 'COHS' or run.GAS_SYS == 'COHSN':
            guessx = float(gas.mCO[-1])
            guessy = float(gas.mS2[-1])
            guessz = float(gas.mO2[-1])
            
            if run.GAS_SYS == 'COHSN':
                guessw = float(gas.mN2[-1])
           
        # Solve using scipy fsolve
        
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            initial_pstep = sys.P_step
            while sys.P_step >= run.DP_MIN:
                try:
                    if run.GAS_SYS =='OH':
                        newguess = optimize.fsolve(oh_eqs, [guessx], args=(run.FE_SYSTEM))
                        return gp.mpfr(newguess[0])
                    
                    elif run.GAS_SYS == "COH":
                        # newguess = optimize.fsolve(coh_eqs, [guessx, guessy], args=(run.FE_SYSTEM))
                        # return gp.mpfr(newguess[0]), gp.mpfr(newguess[1])

                        if melt.graphite_sat == True:
                            newguess = optimize.fsolve(coh_eqs, [guessx], args=(run.FE_SYSTEM, melt.graphite_sat))

                            mCO2 = sl.graphite_fco2(sys.T, sys.P, (O2.Y*newguess[0]*sys.P))/(CO2.Y*sys.P)

                            mCO = (CO2.Y * mCO2)/(sys.K['K2']*CO.Y*(O2.Y*newguess[0]*sys.P)**0.5)

                            return gp.mpfr(newguess[0]), mCO

                        elif melt.graphite_sat == False:
                            newguess = optimize.fsolve(coh_eqs, [guessx, guessy], args=(run.FE_SYSTEM, melt.graphite_sat))                       
                            return gp.mpfr(newguess[0]), gp.mpfr(newguess[1])
                    
                    elif run.GAS_SYS == "SOH":
                        newguess = optimize.fsolve(soh_eqs, [guessx, guessy], args=(run.FE_SYSTEM))
                        return gp.mpfr(newguess[0]), gp.mpfr(newguess[1])
                    
                    elif run.GAS_SYS == 'COHS':
                        # newguess = optimize.fsolve(cohs_eqs, [guessx, guessy, guessz], args=(run.FE_SYSTEM))                       
                        # return gp.mpfr(newguess[0]), gp.mpfr(newguess[1]), gp.mpfr(newguess[2])

                        if melt.graphite_sat == True:
                            newguess = optimize.fsolve(cohs_eqs, [guessy, guessz], args=(run.FE_SYSTEM, melt.graphite_sat))

                            mCO2 = sl.graphite_fco2(sys.T, sys.P, (O2.Y*newguess[1]*sys.P))/(CO2.Y*sys.P)

                            mCO = (CO2.Y * mCO2)/(sys.K['K2']*CO.Y*(O2.Y*newguess[1]*sys.P)**0.5)

                            return mCO, gp.mpfr(newguess[0]), gp.mpfr(newguess[1])

                        elif melt.graphite_sat == False:
                            newguess = optimize.fsolve(cohs_eqs, [guessx, guessy, guessz], args=(run.FE_SYSTEM, melt.graphite_sat))                       
                            return gp.mpfr(newguess[0]), gp.mpfr(newguess[1]), gp.mpfr(newguess[2])
                    
                    elif run.GAS_SYS =='COHSN':

                        if melt.graphite_sat == True:
                            newguess = optimize.fsolve(cohsn_eqs, [guessw, guessy, guessz], args=(run.FE_SYSTEM, melt.graphite_sat))

                            mCO2 = sl.graphite_fco2(sys.T, sys.P, (O2.Y*newguess[2]*sys.P))/(CO2.Y*sys.P)

                            mCO = (CO2.Y * mCO2)/(sys.K['K2']*CO.Y*(O2.Y*newguess[2]*sys.P)**0.5)

                            return gp.mpfr(newguess[0]), mCO, gp.mpfr(newguess[1]), gp.mpfr(newguess[2])

                        elif melt.graphite_sat == False:
                            newguess = optimize.fsolve(cohsn_eqs, [guessw, guessx, guessy, guessz], args=(run.FE_SYSTEM, melt.graphite_sat))                       
                            return gp.mpfr(newguess[0]), gp.mpfr(newguess[1]), gp.mpfr(newguess[2]), gp.mpfr(newguess[3])
                
                except RuntimeWarning as w:
                    message = sys.variable_step()
                    if message is None:
                        sg.set_Y(sys, mols)
                        sys.P_track[-1] = sys.P
                        if sys.graph_unsat_rerun == True:
                            melt.graphite_sat = True  # make sure if previous step was graphite saturated, this is the condition we start with.
                    else:
                        sys.P_step = initial_pstep      # resets pstep for the iron equilibration scenario PL: is this still necessary?
                        print(message)
                        raise RuntimeError(message)               

    # --------------------------------------------------------------------------------------------------------
    # CALCULATION AND PRINT LOOPS
    # --------------------------------------------------------------------------------------------------------

    if run.GAS_SYS == 'OH':

        H2O, O2, H2 = system

        # Recalculate the activity coefficients with the new pressure at the start of each decompression step.
        if sys.P < run.P_START:
            gas.get_ys(system)

        try:
            guessx = jac_newton(run, gas, system)
        except RuntimeError:
            del sys.P_track[-1]     # delete failed pressure step
            msgs.earlyexit(sys, gas, melt, ' Model failed to converge at lowest pressure step.\nData has been written out.\nExiting EVo.')

        if run.FE_SYSTEM == True:
            sys.fe_save(melt, guessx, O2)
        
        gas.mO2.append(guessx)

        print(gas.mO2[-1], sys.P, "End of a pressure step!!!!!!!!!!!!!!!!!!!!!!")

        gas.mH2O.append((1 - gas.mO2[-1]) / (1 + (H2O.Y / (sys.K["K1"] * H2.Y * (gas.mO2[-1] * O2.Y * sys.P) ** 0.5))))
        gas.mH2.append((1 - gas.mO2[-1] - gas.mH2O[-1]))
        empty_lists = [gas.mCO2, gas.mCO, gas.mCH4, gas.mSO2, gas.mS2, gas.mH2S, gas.mN2]
        for list in empty_lists:
            list.append(gp.mpfr(0))

        gas.fo2.append(gp.log(O2.Y*gas.mO2[-1]*sys.P))
        melt.fmq.append(cnvs.fo2_2fmq(gas.fo2[-1], sys.T, sys.P, sys.run.FMQ_MODEL))
        gas.get_fugacity([H2O, H2, 'CO2', 'CO', 'CH4', 'S2', 'SO2', 'H2S', 'N2'], [gas.mH2O[-1], gas.mH2[-1], gas.mCO2[-1], gas.mCO[-1], gas.mCH4[-1], gas.mS2[-1], gas.mSO2[-1], gas.mH2S[-1], gas.mN2[-1]])

        sys.WgT.append(gas.get_WgT(melt, H2O))
        melt.melt_composition(gas, system)
        sys.GvF.append(gas.get_vol_frac(melt))
        sys.rho.append(sys.rho_bulk(melt, gas))
        gas.M.append(cnvs.mean_mol_wt(H2O=gas.mH2O[-1], O2=gas.mO2[-1], H2=gas.mH2[-1]))

        # Check mass is being conserved

        if run.FE_SYSTEM == True:
            o = 'o_tot'
        elif run.FE_SYSTEM == False:
            o = 'o'
        
        for x in ['h', o]:
            if (abs(cnvs.atomicM_calc(sys, melt, gas, x, -1, WgT=sys.WgT[-1]) - sys.atomicM[x])/sys.atomicM[x])*100 > 1e-5 and cnvs.atomicM_calc(sys, melt, gas, x, -1, WgT=sys.WgT[-1]) != 0.0:
                sys.mass_conservation_reset(melt, gas)
                break
        
        if run.RUN_TYPE == 'open':
            gas.open_system(melt, run.LOSS_FRAC)


    # COH system ---------------------------------------------------------------------------------------------
    elif run.GAS_SYS == 'COH':

        H2O, O2, H2, CO, CO2, CH4 = system

        # Recalculate the activity coefficients with the new pressure at the start of each decompression step.
        if sys.P < run.P_START:
            gas.get_ys(system)

        try:
            guessx, guessy = jac_newton(run, gas, system)
        except RuntimeError:
            del sys.P_track[-1]     # delete failed pressure step
            msgs.earlyexit(sys, gas, melt, ' Model failed to converge at lowest pressure step.\nData has been written out.\nExiting EVo.')

        if melt.graphite_sat == True and melt.graph_current <=0:
            melt.graphite_sat = False
            melt.graph_current = 0
            guessx, guessy = jac_newton(run, gas, system)

        if run.C_MODEL == 'eguchi2018':
            # Check graphite saturation
            graph_fCO2 = sl.graphite_fco2(sys.T, sys.P, (O2.Y * guessx * sys.P))

            fCO2 = (sys.K['K2'] * CO.Y * guessy * sys.P * (O2.Y * guessx * sys.P) ** 0.5)

            if float(fCO2) > float(graph_fCO2):
                print('oh no, graphite saturated... gas_fCO2:', float(fCO2), 'graphite_fco2:', float(graph_fCO2))
                melt.graphite_sat == True
        
        if run.FE_SYSTEM == True:
            sys.fe_save(melt, guessx, O2)

        gas.mO2.append(guessx)
        gas.mCO.append(guessy)

        print(gas.mO2[-1], gas.mCO[-1], sys.P, "End of a pressure step!!!!!!!!!!!!!!!!!!!!!!")

        # Append the speciation for the pressure step

        gas.mCO2.append(sys.K['K2'] * CO.Y * gas.mCO[-1] * (O2.Y * gas.mO2[-1] * sys.P) ** 0.5 / CO2.Y)

        gas.mH2O.append((-(1 + ((H2O.Y * sys.P) / (sys.K['K1'] * H2.Y * sys.P * (O2.Y * gas.mO2[-1] * sys.P) ** 0.5))) + ((1 + (
                    (H2O.Y * sys.P) / (
                        sys.K['K1'] * H2.Y * sys.P * (O2.Y * gas.mO2[-1] * sys.P) ** 0.5))) ** 2 - 4 * CO2.Y * gas.mCO2[-1] * (
                            H2O.Y * sys.P) ** 2 / (sys.K['K3'] * CH4.Y * (gas.mO2[-1] * O2.Y * sys.P) ** 2) * -(1 - gas.mO2[-1] - gas.mCO[-1] - gas.mCO2[-1])) ** 0.5) / (2 * CO2.Y * gas.mCO2[-1] * (H2O.Y * sys.P) ** 2 / (sys.K['K3'] * CH4.Y * (gas.mO2[-1] * O2.Y * sys.P) ** 2)))


        gas.mH2.append(H2O.Y * gas.mH2O[-1] / (sys.K['K1'] * H2.Y * (O2.Y * gas.mO2[-1] * sys.P) ** 0.5))

        gas.mCH4.append(CO2.Y*gas.mCO2[-1]*(H2O.Y*gas.mH2O[-1]*sys.P)**2/(sys.K['K3']*CH4.Y*(O2.Y*gas.mO2[-1]*sys.P)**2))

        empty_lists = [gas.mSO2, gas.mS2, gas.mH2S, gas.mN2]
        for list in empty_lists:
            list.append(gp.mpfr(0))

        # Append all extraneous data for the pressure step

        gas.fo2.append(gp.log(O2.Y * gas.mO2[-1] * sys.P))
        melt.fmq.append(cnvs.fo2_2fmq(gp.log10(gp.exp(gas.fo2[-1])), sys.T, sys.P, sys.run.FMQ_MODEL))
        gas.get_fugacity([H2O, H2, CO2, CO, CH4, 'S2', 'SO2', 'H2S', 'N2'], [gas.mH2O[-1], gas.mH2[-1], gas.mCO2[-1], gas.mCO[-1], gas.mCH4[-1], gas.mS2[-1], gas.mSO2[-1], gas.mH2S[-1], gas.mN2[-1]])

        sys.WgT.append(gas.get_WgT(melt, system))
        melt.melt_composition(gas, system)
        sys.GvF.append(gas.get_vol_frac(melt))
        sys.rho.append(sys.rho_bulk(melt, gas))
        gas.M.append(cnvs.mean_mol_wt(H2O=gas.mH2O[-1], O2=gas.mO2[-1], H2=gas.mH2[-1], CO=gas.mCO[-1], CO2=gas.mCO2[-1], CH4=gas.mCH4[-1]))

        # Check mass is being conserved

        if run.FE_SYSTEM == True:
            o = 'o_tot'
        elif run.FE_SYSTEM == False:
            o = 'o'
        
        for x in ['c', o, 'h']:
            if (abs(cnvs.atomicM_calc(sys, melt, gas, x, -1, WgT=sys.WgT[-1]) - sys.atomicM[x])/sys.atomicM[x])*100 > 1e-5 and cnvs.atomicM_calc(sys, melt, gas, x, -1, WgT=sys.WgT[-1]) != 0.0:
                sys.mass_conservation_reset(melt, gas)
                break
        
        if run.RUN_TYPE == 'open':
            gas.open_system(melt, run.LOSS_FRAC)


    elif run.GAS_SYS == 'SOH':

        H2O, O2, H2, S2, SO2, H2S = system

        # Recalculate the activity coefficients with the new pressure at the start of each decompression step.

        if sys.P < run.P_START:
            gas.get_ys(system)

        try:
            guessx, guessy = jac_newton(run, gas, system)
        except RuntimeError:
            del sys.P_track[-1]     # delete failed pressure step
            msgs.earlyexit(sys, gas, melt, ' Model failed to converge at lowest pressure step.\nData has been written out.\nExiting EVo.')

        if run.FE_SYSTEM == True:
            sys.fe_save(melt, guessx, O2)

        gas.mO2.append(guessx)
        gas.mS2.append(guessy)

        print(gas.mO2[-1], gas.mS2[-1], sys.P, "End of a pressure step!!!!!!!!!!!!!!!!!!!!!!")

        gas.mSO2.append((sys.K['K5'] * O2.Y * gas.mO2[-1] * (S2.Y * gas.mS2[-1] * sys.P) ** 0.5) / SO2.Y)

        gas.mH2S.append((1 - gas.mO2[-1] - gas.mS2[-1] - gas.mSO2[-1]) / (1 + (H2S.Y / (sys.K['K1'] * sys.K['K4'] * H2.Y * (S2.Y * gas.mS2[-1] * sys.P) ** 0.5) + (
                (H2S.Y * (O2.Y * gas.mO2[-1]) ** 0.5) / (H2O.Y * sys.K['K4'] * (S2.Y * gas.mS2[-1]) ** 0.5)))))

        gas.mH2.append(H2S.Y * gas.mH2S[-1] / (sys.K['K1'] * sys.K['K4'] * H2.Y * (S2.Y * gas.mS2[-1] * sys.P) ** 0.5))

        gas.mH2O.append(1- gas.mO2[-1] - gas.mH2[-1] - gas.mS2[-1] - gas.mSO2[-1] - gas.mH2S[-1])

        empty_lists = [gas.mCO2, gas.mCO, gas.mCH4, gas.mN2]
        for list in empty_lists:
            list.append(gp.mpfr(0))

        gas.fo2.append(gp.log(O2.Y * gas.mO2[-1] * sys.P))
        melt.fmq.append(cnvs.fo2_2fmq(gas.fo2[-1], sys.T, sys.P, sys.run.FMQ_MODEL))
        gas.get_fugacity([H2O, H2, 'CO2', 'CO', 'CH4', S2, SO2, H2S, 'N2'], [gas.mH2O[-1], gas.mH2[-1], gas.mCO2[-1], gas.mCO[-1], gas.mCH4[-1], gas.mS2[-1], gas.mSO2[-1], gas.mH2S[-1], gas.mN2[-1]])

        #if FeS is saturated as pyrrhotite, solid phase so doesn't generate error, but will mess up the S dissolved in magma.

        sys.WgT.append(gas.get_WgT(melt, system))
        melt.melt_composition(gas, system)
        sys.GvF.append(gas.get_vol_frac(melt))
        sys.rho.append(sys.rho_bulk(melt, gas))
        gas.M.append(cnvs.mean_mol_wt(H2O=gas.mH2O[-1], O2=gas.mO2[-1], H2=gas.mH2[-1], S2=gas.mS2[-1], SO2=gas.mSO2[-1], H2S=gas.mH2S[-1]))

        # Check mass is being conserved

        if run.FE_SYSTEM == True:
            o = 'o_tot'
        elif run.FE_SYSTEM == False:
            o = 'o'
        
        for x in ['s', o, 'h']:
            if (abs(cnvs.atomicM_calc(sys, melt, gas, x, -1, WgT=sys.WgT[-1]) - sys.atomicM[x])/sys.atomicM[x])*100 > 1e-5 and cnvs.atomicM_calc(sys, melt, gas, x, -1, WgT=sys.WgT[-1]) != 0.0:
                sys.mass_conservation_reset(melt, gas)
                break
        
        if run.RUN_TYPE == 'open':
            gas.open_system(melt, run.LOSS_FRAC)

    elif run.GAS_SYS == 'COHS':

        H2O, O2, H2, CO, CO2, CH4, S2, SO2, H2S = system

        # Recalculate the activity coefficients with the new pressure at the start of each decompression step.

        if sys.P < run.P_START:
            gas.get_ys(system)

        try:
            guessx, guessy, guessz = jac_newton(run, gas, system)
        except RuntimeError:
            del sys.P_track[-1]     # delete failed pressure step
            msgs.earlyexit(sys, gas, melt, ' Model failed to converge at lowest pressure step.\nData has been written out.\nExiting EVo.')

        if melt.graphite_sat == True and melt.graph_current <=0:
            melt.graphite_sat = False
            melt.graph_current = 0
            guessx, guessy, guessz = jac_newton(run, gas, system)

        if run.C_MODEL == 'eguchi2018':
            # Check graphite saturation
            graph_fCO2 = sl.graphite_fco2(sys.T, sys.P, (O2.Y * guessz * sys.P))

            fCO2 = (sys.K['K2'] * CO.Y * guessx * sys.P * (O2.Y * guessz * sys.P) ** 0.5)

            if float(fCO2) > float(graph_fCO2):
                print('oh no, graphite saturated... gas_fCO2:', float(fCO2), 'graphite_fco2:', float(graph_fCO2))
        
        if run.FE_SYSTEM == True:
            sys.fe_save(melt, guessz, O2)

        gas.mCO.append(guessx)
        gas.mS2.append(guessy)
        gas.mO2.append(guessz)

        print(gas.mCO[-1], gas.mS2[-1], gas.mO2[-1], sys.P, "End of a pressure step!!!!!!!!!!!!!!!!!!!!!!")

        gas.mCO2.append((sys.K['K2'] * CO.Y * gas.mCO[-1] * (O2.Y * gas.mO2[-1] * sys.P) ** 0.5) / CO2.Y)

        gas.mSO2.append((sys.K['K5'] * O2.Y * gas.mO2[-1] * (S2.Y * gas.mS2[-1] * sys.P) ** 0.5) / SO2.Y)

        a = (CO2.Y*gas.mCO2[-1]*H2O.Y**2)/(sys.K['K3']*CH4.Y*(O2.Y*gas.mO2[-1])**2)
        b = (H2O.Y/(sys.K['K1']*H2.Y*(O2.Y*gas.mO2[-1]*sys.P)**0.5)) + (sys.K['K4']*H2O.Y*(S2.Y*gas.mS2[-1])**0.5/(H2S.Y*(O2.Y*gas.mO2[-1])**0.5)) + 1
        c = -(1-gas.mO2[-1]-gas.mCO2[-1]-gas.mCO[-1]-gas.mSO2[-1]-gas.mS2[-1])

        gas.mH2O.append((-b + (b**2-(4*a*c))**0.5)/(2*a))

        gas.mH2.append((H2O.Y * gas.mH2O[-1]) / (sys.K['K1'] * H2.Y * (O2.Y * gas.mO2[-1] * sys.P) ** 0.5))

        gas.mCH4.append((CO2.Y * gas.mCO2[-1] * (H2O.Y * gas.mH2O[-1]) ** 2) / (sys.K['K3'] * CH4.Y * (O2.Y * gas.mO2[-1]) ** 2))

        gas.mH2S.append((sys.K['K4'] * H2O.Y * gas.mH2O[-1] * (S2.Y * gas.mS2[-1]) ** 0.5) / (H2S.Y * (O2.Y * gas.mO2[-1]) ** 0.5))

        empty_lists = [gas.mN2]
        for list in empty_lists:
            list.append(gp.mpfr(0))
        
        gas.fo2.append(gp.log(O2.Y * gas.mO2[-1] * sys.P))
        melt.fmq.append(cnvs.fo2_2fmq(gas.fo2[-1], sys.T, sys.P, sys.run.FMQ_MODEL))
        gas.get_fugacity([H2O, H2, CO2, CO, CH4, S2, SO2, H2S, 'N2'], [gas.mH2O[-1], gas.mH2[-1], gas.mCO2[-1], gas.mCO[-1], gas.mCH4[-1], gas.mS2[-1], gas.mSO2[-1], gas.mH2S[-1], gas.mN2[-1]])

        # if FeS is saturated as pyrrhotite, solid phase so doesn't generate error, but will mess up the S dissolved in magma.(?)

        sys.WgT.append(gas.get_WgT(melt, system))
        melt.melt_composition(gas, system)
        sys.GvF.append(gas.get_vol_frac(melt))
        sys.rho.append(sys.rho_bulk(melt, gas))
        gas.M.append(cnvs.mean_mol_wt(H2O=gas.mH2O[-1], O2=gas.mO2[-1], H2=gas.mH2[-1], CO=gas.mCO[-1], CO2=gas.mCO2[-1], CH4=gas.mCH4[-1], S2=gas.mS2[-1], SO2=gas.mSO2[-1], H2S=gas.mH2S[-1]))

        # Check mass is being conserved

        if run.FE_SYSTEM == True:
            o = 'o_tot'
        elif run.FE_SYSTEM == False:
            o = 'o'

        for x in ['h', 'c', o, 's']:
            if (abs(cnvs.atomicM_calc(sys, melt, gas, x, -1, WgT=sys.WgT[-1]) - sys.atomicM[x])/sys.atomicM[x])*100 > 1e-5 and cnvs.atomicM_calc(sys, melt, gas, x, -1, WgT=sys.WgT[-1]) != 0.0:
                sys.mass_conservation_reset(melt, gas)
                break
        
        # When running an open system, remove a fraction of the gas phase for the next step.
        if run.RUN_TYPE == 'open':
            gas.open_system(melt, run.LOSS_FRAC)
    
    elif run.GAS_SYS == 'COHSN':

        H2O, O2, H2, CO, CO2, CH4, S2, SO2, H2S, N2 = system

        # Recalculate the activity coefficients with the new pressure at the start of each decompression step.

        if sys.P < run.P_START:
            gas.get_ys(system)

        try:
            guessw, guessx, guessy, guessz = jac_newton(run, gas, system)
        except RuntimeError:
            del sys.P_track[-1]     # delete failed pressure step
            msgs.earlyexit(sys, gas, melt, ' Model failed to converge at lowest pressure step.\nData has been written out.\nExiting EVo.')

        if melt.graphite_sat == True and melt.graph_current <=0:
            melt.graphite_sat = False
            melt.graph_current = 0
            sys.graph_unsat_rerun = True
            guessw, guessx, guessy, guessz = jac_newton(run, gas, system)

        if run.C_MODEL == 'eguchi2018':
            # Check graphite saturation
            graph_fCO2 = sl.graphite_fco2(sys.T, sys.P, (O2.Y * guessz * sys.P))

            fCO2 = (sys.K['K2'] * CO.Y * guessx * sys.P * (O2.Y * guessz * sys.P) ** 0.5)

            if float(fCO2) > float(graph_fCO2):
                print('oh no, graphite saturated... gas_fCO2:', float(fCO2), 'graphite_fco2:', float(graph_fCO2))

            if sys.graph_unsat_rerun == True:
                # reset now the run is complete
                sys.graph_unsat_rerun = False

        if run.FE_SYSTEM == True:
            sys.fe_save(melt, guessz, O2)
        
        gas.mN2.append(guessw)
        gas.mCO.append(guessx)
        gas.mS2.append(guessy)
        gas.mO2.append(guessz)

        print(gas.mN2[-1], gas.mCO[-1], gas.mS2[-1], gas.mO2[-1], sys.P, "End of a pressure step!!!!!!!!!!!!!!!!!!!!!!")

        gas.mCO2.append((sys.K['K2'] * CO.Y * gas.mCO[-1] * (O2.Y * gas.mO2[-1] * sys.P) ** 0.5) / CO2.Y)

        gas.mSO2.append((sys.K['K5'] * O2.Y * gas.mO2[-1] * (S2.Y * gas.mS2[-1] * sys.P) ** 0.5) / SO2.Y)

        a = (CO2.Y*gas.mCO2[-1]*H2O.Y**2)/(sys.K['K3']*CH4.Y*(O2.Y*gas.mO2[-1])**2)
        b = (H2O.Y/(sys.K['K1']*H2.Y*(O2.Y*gas.mO2[-1]*sys.P)**0.5)) + (sys.K['K4']*H2O.Y*(S2.Y*gas.mS2[-1])**0.5/(H2S.Y*(O2.Y*gas.mO2[-1])**0.5)) + 1
        c = -(1-gas.mO2[-1]-gas.mCO2[-1]-gas.mCO[-1]-gas.mSO2[-1]-gas.mS2[-1] - gas.mN2[-1])

        gas.mH2O.append((-b + (b**2-(4*a*c))**0.5)/(2*a))

        gas.mH2.append((H2O.Y * gas.mH2O[-1]) / (sys.K['K1'] * H2.Y * (O2.Y * gas.mO2[-1] * sys.P) ** 0.5))

        gas.mCH4.append((CO2.Y * gas.mCO2[-1] * (H2O.Y * gas.mH2O[-1]) ** 2) / (sys.K['K3'] * CH4.Y * (O2.Y * gas.mO2[-1]) ** 2))

        gas.mH2S.append((sys.K['K4'] * H2O.Y * gas.mH2O[-1] * (S2.Y * gas.mS2[-1]) ** 0.5) / (H2S.Y * (O2.Y * gas.mO2[-1]) ** 0.5))

        gas.fo2.append(gp.log(O2.Y * gas.mO2[-1] * sys.P))
        melt.fmq.append(cnvs.fo2_2fmq(gas.fo2[-1], sys.T, sys.P, sys.run.FMQ_MODEL))
        gas.get_fugacity([H2O, H2, CO2, CO, CH4, S2, SO2, H2S, N2], [gas.mH2O[-1], gas.mH2[-1], gas.mCO2[-1], gas.mCO[-1], gas.mCH4[-1], gas.mS2[-1], gas.mSO2[-1], gas.mH2S[-1], gas.mN2[-1]])

        sys.WgT.append(gas.get_WgT(melt, system))
        melt.melt_composition(gas, system)
        sys.GvF.append(gas.get_vol_frac(melt))
        sys.rho.append(sys.rho_bulk(melt, gas))
        gas.M.append(cnvs.mean_mol_wt(H2O=gas.mH2O[-1], O2=gas.mO2[-1], H2=gas.mH2[-1], CO=gas.mCO[-1], CO2=gas.mCO2[-1], CH4=gas.mCH4[-1], S2=gas.mS2[-1], SO2=gas.mSO2[-1], H2S=gas.mH2S[-1], N2=gas.mN2[-1]))

        # Check mass is being conserved

        if run.FE_SYSTEM == True:
            o = 'o_tot'
        elif run.FE_SYSTEM == False:
            o = 'o'

        for x in ['h', 'c', o, 's', 'n']:
            if (abs(cnvs.atomicM_calc(sys, melt, gas, x, -1, WgT=sys.WgT[-1]) - sys.atomicM[x])/sys.atomicM[x])*100 > 1e-5 and cnvs.atomicM_calc(sys, melt, gas, x, -1, WgT=sys.WgT[-1]) != 0.0:
                sys.mass_conservation_reset(melt, gas)
                break
        
        # When running an open system, remove a fraction of the gas phase for the next step.
        if run.RUN_TYPE == 'open':
            gas.open_system(melt, run.LOSS_FRAC)

    else:
        print("There is no equation for this yet")
