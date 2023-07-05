
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import STAP

def obtainData(symbol, startDate):
    #Read R code, which helps downloading data.
    with open('quantmod_r.r', 'r') as f:
        codeCnt = f.read()
    #Find function "data_collection" in the R code.
    data_collection_in_r = STAP(codeCnt, "data_collection") 
    #Download data of financial product with symbol = "symbol", starting from startDate.
    data_r = data_collection_in_r.data_collection(symbol, startDate) 
    #Enable: Convert ARIMA prediction result to Python pandas dataframe.
    pandas2ri.activate() 
    data_py = robjects.conversion.rpy2py(data_r)
    #Disable: Convert ARIMA prediction result to Python pandas dataframe.
    pandas2ri.deactivate() 
    return data_py

