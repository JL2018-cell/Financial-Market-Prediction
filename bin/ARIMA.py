import csv
import pandas as pd
import numpy as np
import datetime
from rpy2.robjects.packages import STAP #Read R code.
from rpy2.robjects import FloatVector #Convert Python data type to R data Type.
from rpy2.robjects import pandas2ri #Convert R data type to Python data type.
import rpy2.robjects as robjects #Convert R data type to Python data type.
import configparser #Help reading configuration at ../conf/setting.ini

#estimate: 'test': estimtate model based on some available dataset. This helps determine decay factor of sentiment score. 'predict': estimtate model based on all available dataset.
#estimate_ii: determine whether this program should save model or load model after estimation.
def arima_mdl(time_series, frcst_h, estimate, actual_prices, database_name, estimate_ii, prev_prediction_error, prev_prediction, load):
    with open('./Forecast_r_function.r', 'r') as f: #Read program written in R.
        string = f.read()
    forecast_func_in_python= STAP(string, "Forecast_r_function") #Parse R function using STAP.
    config = configparser.ConfigParser()
    config.read('../conf/setting.ini') #Read configuration at ../conf/setting.ini

    #Testing model has completed. Now this model will be used for forecast.
    if (estimate_ii < 0): 
        if (load == 1): #Make final prediction and save it to hard disk.
            forecasted_arima = forecast_func_in_python.Forecast_r_function(time_series = FloatVector(list(time_series)), f_h = frcst_h, estm_ii = estimate_ii, actual_prices = FloatVector(list(actual_prices.iloc[:,0])), prev_prediction_error = prev_prediction_error, prev_prediction = FloatVector(list(prev_prediction)), load = load, load_path = config['DEFAULT']['ARIMA_model_path'])
            #Enable: Convert ARIMA prediction result to Python pandas dataframe.
            pandas2ri.activate()
            arima_pred_py = robjects.conversion.rpy2py(forecasted_arima) 
            #Disable: Convert ARIMA prediction result to Python pandas dataframe.
            pandas2ri.deactivate()
            print(type(arima_pred_py), arima_pred_py)
            #Write forecasted result to hard disk.
            pred_result_pd = pd.DataFrame(arima_pred_py, columns = ['Forecasted value'], index = ['Day' + str(i) for i in range(1, len(arima_pred_py) + 1)])
            pred_result_pd.index.name = "Predicted by ARIMA + Sentiment Analysis Model"
            pred_result_pd.to_csv('../result/forecast_%s.csv' % database_name.split('_')[0])
    #Testing model and estimating model parameters.
    else:
        if (estimate == 'test'): #Estimate model.
            #Divide data into 2 groups:
            #1. y_actual: The most recent 'frcst_h' data items (frcst_h is a number). Used to help calculating residuals. Not involved in model estimation.
            #2. time_series: reamining data items. Used in model estimation.
            y_actual = time_series[len(time_series) - frcst_h: len(time_series)]
            y_actual.index = list(range(1, frcst_h + 1))
            time_series = time_series[0:len(time_series) - frcst_h]
        #If argument: estimate = 'predict', then whole data extracted will be used for estimation.
        #Calling R function, return forecasted result.
        print('ARIMA model estimation:')
        forecasted_arima = forecast_func_in_python.Forecast_r_function(time_series = FloatVector(list(time_series)), f_h = frcst_h, estm_ii = estimate_ii, actual_prices = FloatVector(list(actual_prices)), prev_prediction_error = float(prev_prediction_error), prev_prediction = FloatVector(list(prev_prediction)), load = 0)

        #Convert dataFrame in R to pandas DataFrame in Python.
        pandas2ri.activate()
        arima_pred_py = robjects.conversion.rpy2py(forecasted_arima) 
        pandas2ri.deactivate()

        #Result is: (forecast error, forecast result)
        return (arima_pred_py[0], arima_pred_py[1:len(arima_pred_py)]) 
