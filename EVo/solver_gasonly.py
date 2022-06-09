import constants as cnst
import conversions as cnvs
import gmpy2 as gp
from scipy.optimize import newton
from scipy import linalg
import derivatives_gasonly as diff
import sympy as sp
from sympy import symbols
import numpy as np
import mpmath as mpm
from math import sqrt
import math
from dgs_classes import Molecule
from solvgas import get_K

def gas_only(run, sys, melt, gas, system):
    def jac_newton(run, gas, tol=1e-4, maxiter=1000):

            def F(eqs, x, y):
                return np.array([eqs(x, y)[0], eqs(x, y)[1]])

            def F3(eqs, x, y, z):
                return np.array([eqs(x, y, z)[0], eqs(x, y, z)[1], eqs(x, y, z)[2]])

            def x2jacobian(deriv1, deriv2, equas, guessx, guessy):

                eq1_x, eq1_y = deriv1
                eq2_x, eq2_y = deriv2

                Func = F(equas, guessx, guessy)

                J = np.array(([eq1_x, eq1_y], [eq2_x, eq2_y]))

                det = J[0][0] * J[1][1] - J[0][1] * J[1][0]

                inv_jac = (1 / det) * np.array(([J[-1][-1], -(J[0][-1])], [-(J[-1][0]), J[0][0]]), dtype=object)

                new_guess = np.array([guessx, guessy], dtype=object) - np.dot(inv_jac, Func)  # Returns a 2x1 array of guessx, guessy

                return new_guess[0], new_guess[-1], J  # guessx, guessy, Jacobian

            def x3jacobian(deriv1, deriv2, deriv3, equas, guessx, guessy, guessz):

                eq1_x, eq1_y, eq1_z = deriv1
                eq2_x, eq2_y, eq2_z = deriv2
                eq3_x, eq3_y, eq3_z = deriv3

                Func = F3(equas, guessx, guessy, guessz)

                J = np.array(([eq1_x, eq1_y, eq1_z], [eq2_x, eq2_y, eq2_z], [eq3_x, eq3_y, eq3_z]), dtype=object)

                m1, m2, m3, m4, m5, m6, m7, m8, m9 = J.ravel()

                determinant = m1*m5*m9 + m4*m8*m3 + m7*m2*m6 - m1*m6*m8 - m3*m5*m7 - m2*m4*m9  
                inv_jac = np.array([[m5*m9-m6*m8, m3*m8-m2*m9, m2*m6-m3*m5],
                            [m6*m7-m4*m9, m1*m9-m3*m7, m3*m4-m1*m6],
                            [m4*m8-m5*m7, m2*m7-m1*m8, m1*m5-m2*m4]])/determinant         
                
                dot = np.dot(inv_jac, Func)  # To get the arrays as 2 3x1 columns
                new_guess = np.array([guessx, guessy, guessz]) - dot  # Returns a 3x1 array of guessx, guessy, guessz
                return new_guess[0], new_guess[1], new_guess[2], J  # guessx, guessy, guessz, Jacobian

            if run.GAS_SYS == 'COHS' and sys.OCS==True:

                # Save the first guess as the result from the previous pressure step
                prevx = gas.mCO[-1]
                prevy = gas.mS2[-1]
                prevz = gas.mO2[-1]
                
                for iter in range(maxiter):
                    deriv1, deriv2, deriv3 = diff.cohs_ocs_o(sys, gas, system, prevx, prevy, prevz), diff.cohs_ocs_h(sys, gas, system, prevx, prevy, prevz), diff.cohs_ocs_s(sys, gas, system, prevx, prevy, prevz)

                    # Run the Newton Rapheson step and return the guess values and the Jacobian matrix
                    guessx, guessy, guessz, J = x3jacobian(deriv1, deriv2, deriv3, cohs_ocs_eqs, prevx, prevy, prevz)

                    # If there are NaN values they are returned to trigger a variable pressure step change in outer loops
                    if math.isnan(guessx) == True or math.isnan(guessy) == True or math.isnan(guessz) == True:
                        return guessx, guessy, guessz

                    if guessx < 0 or guessy < 0 or guessz < 0:
                        exit('OCS step has returned a negative value')

                    # Test for convergence, reached when the difference on each variable is 0.00001%
                    if (abs(prevx - guessx) / prevx) * 100 < tol and (abs(prevy - guessy) / prevy) * 100 < tol and (abs(prevz - guessz) / prevz) * 100 < tol:
                        return guessx, guessy, guessz

                    # Redefine the initial guess as the result of the last loop ready to run again.
                    prevx = guessx
                    prevy = guessy
                    prevz = guessz

                    print(prevx, prevy, prevz)

                msg = "Failed to converge after %d iterations, values are %s" % (maxiter, [guessx, guessy, guessz, sys.P])
                raise RuntimeError(msg)

    def cohs_ocs_eqs(guessx=1., guessy=1., guessz=1.):
            # x = mCO, y = mS2, z = mO2

            def eq_cohs_o(x, y, z):
                # x = mCO, y = mS2, z = mO2

                mCO2 = (sys.K['K2'] * CO.Y * x * (O2.Y * z * sys.P) ** 0.5) / CO2.Y

                mSO2 = (sys.K['K5'] * O2.Y * z * (S2.Y * y * sys.P) ** 0.5) / SO2.Y

                a = (CO2.Y*mCO2*H2O.Y**2)/(sys.K['K3']*CH4.Y*(O2.Y*z)**2)
                b = (H2O.Y/(sys.K['K1']*H2.Y*(O2.Y*z*sys.P)**0.5)) + (sys.K['K4']*H2O.Y*(S2.Y*y)**0.5/(H2S.Y*(O2.Y*z)**0.5)) + 1
                c = -(1-z-mCO2-x-mSO2-y)

                mH2O = (-b + (b**2-(4*a*c))**0.5)/(2*a)        
                
                mCH4 = (CO2.Y * mCO2 * (H2O.Y * mH2O) ** 2) / (sys.K['K3'] * CH4.Y * (O2.Y * z) ** 2)

                mH2S = (sys.K['K4']*H2O.Y*mH2O*(S2.Y*y)**0.5)/(H2S.Y*(O2.Y*z)**0.5)
                
                mOCS = (CO2.Y * mCO2 * H2S.Y * mH2S) / (sys.K['K6'] * OCS.Y * H2O.Y * mH2O)

                C = (gas.atomicM['c'] / cnst.m['c']) / (x + mCO2 + mCH4 + mOCS)

                return (C*(2*z + 2*mSO2 + mH2O + x + 2*mCO2 + mOCS)) - (gas.atomicM['o']/cnst.m['o'])

            def eq_cohs_h(x, y, z):
                # x = mCO, y = mS2, z = mO2

                mCO2 = (sys.K['K2'] * CO.Y * x * (O2.Y * z * sys.P) ** 0.5) / CO2.Y

                mSO2 = (sys.K['K5'] * O2.Y * z * (S2.Y * y * sys.P) ** 0.5) / SO2.Y

                mH2O = (-(1 + ((sys.K['K4'] * H2O.Y * (S2.Y * y) ** 0.5) / (H2S.Y * (O2.Y * z) ** 0.5)) + (H2O.Y / (sys.K['K1'] * H2.Y * (O2.Y * z * sys.P) ** 0.5))) + ((1 + ((sys.K['K4'] * H2O.Y * (S2.Y * y) ** 0.5) / (H2S.Y * (O2.Y * z) ** 0.5)) + (
                                            H2O.Y / (sys.K['K1'] * H2.Y * (O2.Y * z * sys.P) ** 0.5))) ** 2 - 4 * (CO2.Y * mCO2 * (H2O.Y ** 2)) / (sys.K['K3'] * CH4.Y * (O2.Y * z) ** 2) * -(
                                        1 - x - y - z - mCO2 - mSO2)) ** 0.5) / (2 * (CO2.Y * mCO2 * (H2O.Y ** 2)) / (sys.K['K3'] * CH4.Y * (O2.Y * z) ** 2))

                mH2 = (H2O.Y * mH2O) / (sys.K['K1'] * H2.Y * (O2.Y * z * sys.P) ** 0.5)

                mCH4 = (CO2.Y * mCO2 * (H2O.Y * mH2O) ** 2) / (sys.K['K3'] * CH4.Y * (O2.Y * z) ** 2)

                mH2S = (sys.K['K4'] * H2O.Y * mH2O * (S2.Y * y) ** 0.5) / (H2S.Y * (O2.Y * z) ** 0.5)
                
                mOCS = (CO2.Y * mCO2 * H2S.Y * mH2S) / (sys.K['K6'] * OCS.Y * H2O.Y * mH2O)

                C = (gas.atomicM['c'] / cnst.m['c']) / (x + mCO2 + mCH4 + mOCS)

                return (C * (mH2O + mH2 + mH2S + 2*mCH4 + mOCS)) - (gas.atomicM['h']/(2*cnst.m['h']))

            def eq_cohs_s(x, y, z):

                # x = mCO, y = mS2, z = mO2

                mCO2 = (sys.K['K2']*CO.Y*x*(O2.Y*z*sys.P)**0.5)/CO2.Y

                mSO2 = (sys.K['K5']*O2.Y*z*(S2.Y*y*sys.P)**0.5)/SO2.Y

                a = (CO2.Y*mCO2*H2O.Y**2)/(sys.K['K3']*CH4.Y*(O2.Y*z)**2)
                b = (H2O.Y/(sys.K['K1']*H2.Y*(O2.Y*z*sys.P)**0.5)) + (sys.K['K4']*H2O.Y*(S2.Y*y)**0.5/(H2S.Y*(O2.Y*z)**0.5)) + 1
                c = -(1-z-mCO2-x-mSO2-y)

                mH2O = (-b + (b**2-(4*a*c))**0.5)/(2*a)         

                mCH4 = (CO2.Y*mCO2*(H2O.Y*mH2O)**2)/(sys.K['K3']*CH4.Y*(O2.Y*z)**2)

                mH2S = (sys.K['K4']*H2O.Y*mH2O*(S2.Y*y)**0.5)/(H2S.Y*(O2.Y*z)**0.5)

                mOCS = (CO2.Y * mCO2 * H2S.Y * mH2S) / (sys.K['K6'] * OCS.Y * H2O.Y * mH2O)

                C = (gas.atomicM['c'] / cnst.m['c']) / (x + mCO2 + mCH4 + mOCS)  # NONE OF THE -1 STUFF :(

                return C*(mSO2 + mH2S + 2*y + mOCS) - (gas.atomicM['s']/cnst.m['s'])

            return eq_cohs_o(guessx, guessy, guessz), eq_cohs_h(guessx, guessy, guessz), eq_cohs_s(guessx, guessy, guessz)
    
    if run.GAS_SYS == 'COHS' and sys.OCS == True:

        H2O, O2, H2, CO, CO2, CH4, S2, SO2, H2S = system      
                    
        OCS = Molecule(run, sys, melt, 'OCS')

        system = H2O, O2, H2, CO, CO2, CH4, S2, SO2, H2S, OCS  # Add OCS into the molecules being tracked
        
        sys.K = get_K(sys, system)
        gas.get_ys(system)

        gas.get_atomic_mass(run, sys)

        guessx, guessy, guessz = jac_newton(run, gas)

        gas.mCO.append(guessx)
        gas.mS2.append(guessy)
        gas.mO2.append(guessz)

        gas.mCO2.append((sys.K['K2'] * CO.Y * gas.mCO[-1] * (O2.Y * gas.mO2[-1] * sys.P) ** 0.5) / CO2.Y)

        gas.mSO2.append((sys.K['K5'] * O2.Y * gas.mO2[-1] * (S2.Y * gas.mS2[-1] * sys.P) ** 0.5) / SO2.Y)

        a = (CO2.Y*gas.mCO2[-1]*H2O.Y**2)/(sys.K['K3']*CH4.Y*(O2.Y*gas.mO2[-1])**2)
        b = (H2O.Y/(sys.K['K1']*H2.Y*(O2.Y*gas.mO2[-1]*sys.P)**0.5)) + (sys.K['K4']*H2O.Y*(S2.Y*gas.mS2[-1])**0.5/(H2S.Y*(O2.Y*gas.mO2[-1])**0.5)) + 1
        c = -(1-gas.mO2[-1]-gas.mCO2[-1]-gas.mCO[-1]-gas.mSO2[-1]-gas.mS2[-1])

        gas.mH2O.append((-b + (b**2-(4*a*c))**0.5)/(2*a))

        gas.mH2.append((H2O.Y * gas.mH2O[-1]) / (sys.K['K1'] * H2.Y * (O2.Y * gas.mO2[-1] * sys.P) ** 0.5))

        gas.mCH4.append((CO2.Y * gas.mCO2[-1] * (H2O.Y * gas.mH2O[-1]) ** 2) / (sys.K['K3'] * CH4.Y * (O2.Y * gas.mO2[-1]) ** 2))

        gas.mH2S.append((sys.K['K4'] * H2O.Y * gas.mH2O[-1] * (S2.Y * gas.mS2[-1]) ** 0.5) / (H2S.Y * (O2.Y * gas.mO2[-1]) ** 0.5))

        gas.mOCS.append((CO2.Y * gas.mCO2[-1] * H2S.Y * gas.mH2S[-1]) / (sys.K['K6'] * OCS.Y * H2O.Y * gas.mH2O[-1]))

        gas.fo2.append(gp.log(O2.Y * gas.mO2[-1] * sys.P))
        melt.fmq.append(melt.fmq[-1])
        gas.get_fugacity([H2O, H2, CO2, CO, CH4, S2, SO2, H2S], [gas.mH2O[-1], gas.mH2[-1], gas.mCO2[-1], gas.mCO[-1], gas.mCH4[-1], gas.mS2[-1], gas.mSO2[-1], gas.mH2S[-1]])

        melt.h2o.append(melt.h2o[-1])
        melt.h2.append(melt.h2[-1])
        melt.co2.append(melt.co2[-1])
        melt.so2.append(melt.so2[-1])
        melt.h2s.append(melt.h2s[-1])

        sys.WgT.append(sys.WgT[-1])
        sys.GvF.append(sys.GvF[-1])
        sys.rho.append(sys.rho[-1])

    else:
        print('This gas only system hasn\'t been coded yet')