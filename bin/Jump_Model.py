#ATTENTION: Need to run in administrator's cmd!
#Receive time series data from main.py instead of retrieving time series in this function.
#adapt to sampling scheme. Add save model features.
import matlab
import matlab.engine
import matplotlib.pyplot as plt
import Detect_Jumps
import numpy as np
import read_JmpMdl_data
import pandas as pd
import configparser

def matlab_simul(time_series, forecast_horizon, startStockPrice, simul_Num, estimate_ii, actual_prices, database_name, prev_prediction_error, prev_prediction, load = 0):
    eng = matlab.engine.start_matlab() #Start Matlab application.
    config = configparser.ConfigParser() #Read configuration setting.
    config.read('../conf/setting.ini') #Setting is stored in this path.
    #If main.py tells Jump Model is the best model of forecast, then, load model saved in ../temp/ and forecast.
    if (estimate_ii < 0):
        if (load == 1):
            #Read Path of storing parameters of Jump Model
            parameters = read_JmpMdl_data.read_JmpMdl(config['DEFAULT']['read_JmpMdl'])
            #Read parameter "Standard deviation" of Jump Model.
            StdDev = parameters['StdDev']
            #Read parameter "drift" of Jump Model.
            drift = parameters['drift']
            #Read parameter "mean of Poisson Distribution" of Jump Model.
            Poi_rate = parameters['Poi_rate']
            #Read parameters "rate of Exponential Distribution" of Jump Model.
            Expn_rate = parameters['Expn_rate'] 
            #Call Matlab function to execute Jump Model.
            ret = eng.Jump_Model_run(0, matlab.double([Poi_rate]), matlab.double([drift]), matlab.double([StdDev]), matlab.double([Expn_rate]), forecast_horizon, simul_Num, matlab.double([startStockPrice])) 
            #Save final prediction to hard disk.
            index = ['Day' + str(i) for i in range(1, len(np.array(ret._data)) + 1)]
            ret_pd = pd.DataFrame(np.array(ret._data), index = index, columns = ['Forecasted value'])
            ret_pd.index.name = "Predicted by Jump Model"
            ret_pd.to_csv('../result/forecast_%s.csv' % database_name)
            print('Jump Model: prediction result is saved.')
    else:
        #Calculate standard deviation of data.
        StdDev = time_series.pct_change().dropna().var()**0.5 
        #Calculate mean/average of data.
        drift = time_series.pct_change().dropna().mean()
        #Estimate frequency of jumps and magnitude of jumps by Poisson Distribution and Exponential Distribution respectively.
        [Poi_rate, Expn_rate] = Detect_Jumps.estm_rates(matlab.double(list(time_series.pct_change().dropna())))
        #Call Matlab function to execute Jump Model.
        ret = eng.Jump_Model_run(0, matlab.double([Poi_rate]), matlab.double([drift]), matlab.double([StdDev]), matlab.double([Expn_rate]), forecast_horizon, simul_Num, matlab.double([startStockPrice]))
        #This is first time of model estimation. So, we need to save estimation result first.
        if (estimate_ii == 0):
            f = open(config['DEFAULT']['read_JmpMdl'], 'w') #Save parameters of model.
            f.write('StdDev ' + str(StdDev) + '\n')
            f.write('drift ' + str(drift) + '\n')
            f.write('Poi_rate ' + str(Poi_rate) + '\n')
            f.write('Expn_rate ' + str(Expn_rate) + '\n')
            f.write('simul_Num ' + str(simul_Num) + '\n')
            f.close()
            #Format of calculaion result: (prediction squared error, prediction values)
            return (np.linalg.norm(actual_prices.to_numpy().reshape(1,-1)[0] -  np.array(ret._data))**2, np.array(ret._data))
        #This is not the first time of estimation. So, we need to compare previous prediction error dn this prediction error. 
        else: #estimate_ii > 0
            this_prediction_error = np.linalg.norm(actual_prices.to_numpy().reshape(1,-1)[0]-np.array(ret._data))**2
            #If former error > later error, then, save this model, and return this forecast error and forecast values.
            if (this_prediction_error < prev_prediction_error):
                f = open('JmpMdl.dat', 'w') #Save parameters of model.
                f.write('StdDev ' + str(StdDev) + '\n')
                f.write('drift ' + str(drift) + '\n')
                f.write('Poi_rate ' + str(Poi_rate) + '\n')
                f.write('Expn_rate ' + str(Expn_rate) + '\n')
                f.write('simul_Num ' + str(simul_Num) + '\n')
                f.close()
                print('This estimate is better. Save Jump model.')
                #Format: (prediction squared error, prediction values)
                return (this_prediction_error, np.array(ret._data))
            #If former error < later error, then, 
            #former forecast is better than this forecast. 
            #So, model estimated here is discarded.
            else: 
                print('Previous estimate is better. Save Jump Model.')
                #Format of calculation result: (prediction squared error, prediction values)
                return (prev_prediction_error, prev_prediction)
