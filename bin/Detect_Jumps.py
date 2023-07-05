#ATTENTION: Need to run in administrator's command prompt!

#Caclulate Poisson rate and Exponential distribution parameter
#from a historical time series.

import numpy as np
import matlab
import matlab.engine
import matplotlib.pyplot as plt

def estm_rates(data):
    eng = matlab.engine.start_matlab() #Start Matlab application program.

    #Call Matlab function "Jump_Statistics" to estimate frequency of Jumps and 
    #magnitude of jumps with Poisson Distribution and 
    #Exponential Distribution respectively.
    ret = eng.Jump_Statistics(data) 

    print('Type of ret =', type(np.array(ret._data)))
    print('ret =', np.array(ret._data)[0])

    #Convert from Matlab matrix to Python array.
    return [i for i in np.array(ret._data)]
