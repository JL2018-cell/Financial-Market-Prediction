import pandas as pd
import numpy as np
from rpy2.robjects.packages import STAP
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

def optmzWeight(netSentiment, pred_data):
    data = pd.DataFrame({'y': pred_data, 'x': [netSentiment for i in range(len(pred_data))]}) #Organise 2 arguments: netSentiment, pred_data into pandas DataFrame.
    #Read R code.
    with open('./weight_optmz.r', 'r') as f:
        codeCnt = f.read()
    forecast_func_in_python= STAP(codeCnt, "fndWght")

    with localconverter(robjects.default_converter + pandas2ri.converter):
        #Convert Python pandas dataframe to R dataframe.
        r_from_pd_df = robjects.conversion.py2rpy(data)
    #Call R function.
    forecasted_arima=forecast_func_in_python.fndWght(r_from_pd_df)
    #Convert form R vector (length = 1) to Python number.
    return list(forecasted_arima)[0]
