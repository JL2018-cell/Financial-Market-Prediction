print('The system is loading different libraries. Please wait...')
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import matlab #Use Matlab programming language.
import matlab.engine #Use Matlab programming language.
import preliminary_NLP #local file, natural language processing.
import further_NLP #local file, natural language processing.
import ARIMA #local file, autoregressive integrated moving average.
import Find_weight #local file, 
import web_scraper #local file, obtain textual data by web-scrapping.
import sql_operation #local file, obtain data from local SQL database.
import Jump_Model #local file, forecast with Geometric Brownian Motion.
import LSTM #local file, forecast with Long short-Term Memory.
import sampling_scheme #local file, remove some data items before forecasting.
import show_msg #local file, display message box.
import configparser #read configurations located at ../config/setting.ini

def main():
    #Define some commonly used variables, settings in this Financial Data Forecaster.
    #The meaning of variables is explained in ../config/setting.ini
    config = configparser.ConfigParser()
    config.read('../conf/setting.ini')
    sampling_rate = float(config['DEFAULT']['sampling_rate'])
    sampling_trial_num = int(config['DEFAULT']['sampling_trial_num'])
    ratio = float(config['DEFAULT']['ratio'])
    frcst_h = int(config['DEFAULT']['frcst_hz'])
    simul_Num = int(config['DEFAULT']['simul_Num'])
    numOfTest = int(config['DEFAULT']['numOfTest'])
    plot_path = config['DEFAULT']['save_plot_path']

    #If a database is missing, then, 
    #this program will collect data equal or later than this date.
    database_start_date = datetime.datetime(2019, 12, 31) 

    #If multiple financial data series are forecasted AND user specify the program to forcast from today, 
    #then, this program will start forecasting from the least recent latest date 
    #selected SQL databases.
    database_latest_date = datetime.datetime.today().date()

    #Create SQL database for new financial product if not.
    for item in config.sections():
        sql_operation.create_sql_table(config[item]['database_name'], config[item]['symbol'], database_start_date)

    #Set up new SQL database and fill the empty database with data if:
    #target database is missing.
    for item in config.sections(): 
        tmp_date = sql_operation.latest_date(config[item]['database_name'])
        if (tmp_date != None):
            if (database_latest_date - tmp_date > datetime.timedelta(seconds = 0)):
                database_latest_date = tmp_date

    #Ask user date of forecast.
    timeprefer = input('Forecast according to latest data or specified date? ([L]atest / [S]pecified)')
    #timeprefer = 's' #Debug
    #User wants to forecast from latest date.
    if (timeprefer == 'L' or timeprefer == 'l'):
        #Display message
        show_msg.show_msg('Stage 0: web-scrapping')
        #Update stock data in SQL database.
        for item in config.sections():
            database = config[item]['database_name']
            symbol = config[item]['symbol']
            #If database is not updated.
            if (not sql_operation.info_updateOrNot(database)):
                print('Your database is not updated! let me help you update first.')
                sql_operation.update_sql_price(database, symbol, database_start_date)
        print('Starts web-scrapping... Please ignore the output from web-scrapper in the following.')
        #Update text data in SQL database.
        for item in config.sections():
            database = config[item]['database_name']
            symbol = config[item]['symbol']
            keyword = config[item]['keyword']
            region = config[item]['region']
            #Latest date shown in database, format = Python datetime
            requiredDate = sql_operation.latest_date(database)
            web_scraper.scrap_news(database, keyword, region, requiredDate)
            print('Finish web-scrapping and save the data to %s!' % database)
        print('Finally, web-scrapping is finished!')
    #User wants to forecast from specified date.
    else:
        #Used to check if user input is valid.
        invalidInput = True
        while(invalidInput):
            dateprefer = input('What is the date you would like to start forecast? Format: yyyy-mm-dd')
            #dateprefer = '2021-11-8' #Debug
            requiredDate = datetime.datetime.strptime(dateprefer, '%Y-%m-%d')
            #If user input date is too early.
            if (requiredDate < database_start_date + datetime.timedelta(days = 7)):
                print("Sorry, you should input date later than", (database_start_date + datetime.timedelta(days = 7)).date())
            #If user input date > today date.
            elif (requiredDate.date() > database_latest_date):
                print("Sorry, you should input date earlier than", database_latest_date)
            else:
                print("Okay.")
                invalidInput = False #Break while loop.

    #Set parameter of forecasting
    #template sentences used for NLP training, Format = pandas dataframe
    txtSamples = sql_operation.extract_sentiment()
    #Record actual prices in the most recent 3 'frcst_h' days, so that squared error of prediction can be calculated.
    actual_prices = list()
    #Record average squared error of prediction done by ARIMA.
    SqErrARIMA = 0 
    #Record average squared error of prediction done by Jump Model.
    SqErrJmpMdl = 0
    #Record average squared error of prediction done by LSTM Model.
    SqErrlstm = 0 
    #Initialize Previous prediction = [infinity, ...,infinity].
    prev_prediction = pd.Series([float('inf') for i in range(frcst_h)]) 
    #Initialize Previous prediction error = infinity.
    prev_prediction_error = float('inf') 

    #Display message.
    show_msg.show_msg('Stage 1: Testing models') 
    for item in config.sections():
        #database name.
        database = config[item]['database_name'] 
        #Symbol of financial sceurity e.g. symbol of Tesla is "TSLA"
        symbol = config[item]['symbol'] 
        show_msg.show_msg('Stage 1: Forecast ' + database)
        #Set up start date and end dates of data to be evaluated.
        #This is placed in loop because latest trading days for different stocks may be different.
        #User wants to forecast from latest date/today.
        if (timeprefer == 'L' or timeprefer == 'l'):
            #Take past 1 year data for estimation.
            startDate = datetime.datetime.today() - datetime.timedelta(days = 365)
            endDate = datetime.datetime.today()
        #User wants to forecast from specified date.
        else:
            #Take past 1 year data for estimation.
            startDate = requiredDate - datetime.timedelta(days = 365)
            endDate = requiredDate
        #For displaying messages below only.
        nearestDate = sql_operation.nearest_date_before(database, endDate)
        print('The nearest date that database', database, 'has relevant data is', nearestDate.date(), '(Format: yyyy-mm-dd)')
        nearestStartDate = sql_operation.last_n_day(database, startDate, frcst_h * numOfTest)
        #Available data items in database is less than 365 days.
        if (nearestStartDate is None):
            #Take the earliest date that the database has.
            nearestStartDate = sql_operation.earliest_date(database)
            #Set up new estimation sample range.
            startDate = sql_operation.last_n_day(database, nearestStartDate, frcst_h * numOfTest, 0)
        #Find the past 'frcst_h' th trading day.
        nearestEndDate = sql_operation.last_n_day(database, endDate, frcst_h * numOfTest) 
        #Change date to string format
        strNearestStartDate = str(nearestStartDate).split()[0] 
        #Change date to string format
        strNearestEndDate = str(nearestEndDate).split()[0]

        #Show message
        show_msg.show_msg('Stage 1: Estimating data of ' + database)
        #Test for numOfTest times
        for i in range(numOfTest): 
            #Extract text data 'news' from SQL.
            news = sql_operation.extract_text(database, 'news', nearestEndDate) 
            #Different news are separated by string = '\n----\n'. 
            #It is necessary to place each piece of text into different posiiton
            #in a Python list.
            news_text = news.split('\n----\n') 
            comments = sql_operation.extract_text(database, 'comments', nearestEndDate) 
            #Extract text data 'comments' from SQL.
            comments_text = comments.split('\n----\n') 
            #Combine 2 types of text together.
            comments = comments_text + news_text 
            #Remove comments that are empty strings.
            comments = list(filter(lambda comment: len(comment), comments)) 
            #Do natural language processing if both training data and text data from database are not null.
            if ((not txtSamples.empty) and (len(comments) > 0)): 
                print('Text training data and web-scrapped text in database', database, 'are present.')
                print('Training Natural Language Processing models. Please wait...')
                #Return pandas DataFrame, column 1 = sentiment score (1 = positive, -1 = negative), column 2 = corresponding phrases.
                sentmScores = preliminary_NLP.findEmotion(txtSamples, comments, ratio) 
                targetEntity = config[item]['keyword'] #Company name
                #Further classify how positive this news is.
                sentmRatesPov = further_NLP.rateEmotionTrain(txtSamples, sentmScores[sentmScores['Sentiment'] == 1]['Phrase'], targetEntity) 
                #Further classify how negative this news is.
                sentmRatesNeg = further_NLP.rateEmotionTrain(txtSamples, sentmScores[sentmScores['Sentiment'] == -1]['Phrase'], targetEntity) 
                netSentiment = (sentmRatesPov + sentmRatesNeg) / len(comments)
                print('Type = ', type(netSentiment))
                print('netSentiment = ', netSentiment)
            #No text available for sentiment analysis.
            else: 
                print('Not enough text training data and/or web-scrapped text in database', database)
                print('Set sentiment score = 0')
                #Set sentiment score = 0
                netSentiment = 0 

            #Initialize prediction error of ARIMA + sentiment analysis.
            mdl_pred_error = 0
            mdl_pred_info = pd.Series([float('inf') for i in range(frcst_h)])
            #Store average forecast result after sampling.
            mdl_pred_info_avg = pd.Series([0 for i in range(frcst_h)]) 
            #Initialize prediction error of Jump Model.
            jmpMdlErr = 0 
            jmpMdlPred = pd.Series([float('inf') for i in range(frcst_h)])
            #Store average forecast result after sampling.
            jmpMdlPred_avg = pd.Series([0 for i in range(frcst_h)]) 
            #Initialize prediction error of LSTM.
            lstmErr = 0 
            lstmPred = pd.Series([float('inf') for i in range(frcst_h)])
            #Store average forecast result after sampling.
            lstmPred_avg = pd.Series([0 for i in range(frcst_h)]) 

            #Obtain numerical data/historical price data.
            time_series_data = sql_operation.extract_all_numeric_data(database, strNearestStartDate, strNearestEndDate) 
            actual_prices.append(sql_operation.extract_data_num(database, strNearestEndDate, frcst_h))
            for ii in range(sampling_trial_num):
                #if (ii < sampling_trial_num):
                #Remove a small part of data. It is treated as noise.
                time_series = sampling_scheme.sampling(time_series_data, 1 - sampling_rate)
                time_series.sort_index()
                #Method 1: ARIMA + natural language processing
                print('Testing', symbol, 'from', strNearestStartDate, 'to', strNearestEndDate, '...')
                #If sentiment score != 0.
                if (netSentiment != 0): 
                    mdl_pred_error, mdl_pred_info = ARIMA.arima_mdl(time_series = time_series['Close'], frcst_h = frcst_h, estimate = 'test', actual_prices = actual_prices[i].iloc[:,0], database_name = database, estimate_ii = ii, prev_prediction_error = mdl_pred_error, prev_prediction = mdl_pred_info, load = 0)
                    #Calculate prediction error of each day.
                    mdl_pred_error_series = pd.Series(actual_prices[len(actual_prices) - 1].iloc[:,0]) - mdl_pred_info 
                    #optimization requires varying numbers, Otherwise, there is runtime error.
                    if (mdl_pred_error_series.var() != 0.0 and netSentiment != 0): 
                    #Find weight of sentiment score such that prediciton error is further miminized.
                        exp_weight_r = Find_weight.optmzWeight(netSentiment, mdl_pred_error_series) 
                        #Convert form Floatvector in R to float in Python.
                        exp_weight = list(exp_weight_r)[0] 
                    #Fluctuation of predicted value perfectly match actual data. 
                    #As a result, sentiment analysis no needs to help anymore.
                    else: 
                        exp_weight = 0
                    print('Type = ', type(exp_weight))
                    print('exp_weight = ', exp_weight)
                #If sentiment score = 0.
                else: 
                    exp_weight = 0
                mdl_pred_error, mdl_pred_info = ARIMA.arima_mdl(time_series = time_series['Close'], frcst_h = frcst_h, estimate = 'predict', actual_prices = actual_prices[i].iloc[:,0], database_name = database, estimate_ii = ii, prev_prediction_error = mdl_pred_error, prev_prediction = mdl_pred_info, load = 0)
                print('Type = ', type(mdl_pred_info))
                print('mdl_pred_info = ', mdl_pred_info)
                #Adjust ARIMA model prediction with sentiment score.
                mdl_pred_info += np.array([netSentiment*(exp_weight**(1+i)) for i in range(frcst_h)]) 
                #Store average forecast result of sampling scheme.
                mdl_pred_info_avg += mdl_pred_info 
                #Calculate accumulated prediction error.
                SqErrARIMA += mdl_pred_error 
        
                #Method 2: Jump Model
                #Call Jump Model function to predict price.
                jmpMdlErr, jmpMdlPred = Jump_Model.matlab_simul(time_series = time_series['Close'], forecast_horizon = frcst_h, startStockPrice = actual_prices[len(actual_prices) - 1].iloc[0, 0], simul_Num = simul_Num, estimate_ii = ii, actual_prices = actual_prices[i], database_name = database, prev_prediction_error = jmpMdlErr, prev_prediction = jmpMdlPred, load = 0) 
                #Store average forecast result of sampling scheme.
                jmpMdlPred_avg += jmpMdlPred 
                #Calculate accumulated prediction error.
                SqErrJmpMdl += jmpMdlErr 
    
                #Method 3: Machine learning, LSTM
                lstmErr, lstmPred = LSTM.LSTM(time_series = time_series, frcst_hz = frcst_h, actual_prices = actual_prices[len(actual_prices) - 1], database_name = database, estimate_ii = ii, prev_prediction_error = lstmErr, prev_prediction = lstmPred, load = 0) #Call LSTM function to predict movement of price.
                #Calculate accumulated prediction error.
                lstmPred_avg += lstmPred 

            #Plot graph of forecasted value.
            time_series_np = np.array(time_series['Close'].sort_index()).reshape(1,-1)[0]
            #Take average of forecasted result.
            mdl_pred_info_avg /= sampling_trial_num 
            jmpMdlPred_avg /= sampling_trial_num
            lstmPred_avg /= sampling_trial_num
            lstmPred_plot = np.concatenate((time_series_np, lstmPred_avg), axis = 0)
            plt.plot(lstmPred_plot[-9*frcst_h:], label = 'LSTM')
            jmpMdlPred_plot = np.concatenate((time_series_np, jmpMdlPred_avg), axis = 0)
            plt.plot(jmpMdlPred_plot[-9*frcst_h:], label = 'Jump Model')
            mdl_pred_plot = np.concatenate((time_series_np, mdl_pred_info_avg), axis = 0)
            plt.plot(mdl_pred_plot[-9*frcst_h:], label = 'ARIMA')
            time_series_np = np.concatenate((time_series_np[-8*frcst_h:], actual_prices[-1]['Close']), axis = 0)
            plt.plot(time_series_np, label = 'Actual') #historical prices
            plt.title(database + ': Forecasted Values of Different Methods')
            plt.legend(shadow=True)
            plt.xticks([len(time_series_np)//2, len(time_series_np) - 1], [str(actual_prices[-1].index[0]).split()[0], str(actual_prices[-1].index[-1]).split()[0]])
            plt.savefig(plot_path + database + '_test' + str(i) + '.png', dpi=200)
            plt.clf()

            #Actual prices in the forecast horizon.
            Actual = np.array(actual_prices[-1]).reshape(1,-1)[0]
            #Predicted values of LSTM Model.
            Predicted = lstmPred_plot[-5:].reshape(1,-1)[0]
            #Dates of the forecast horizon.
            index = actual_prices[-1].index
            #Save tested prediction result.
            save_path = config['DEFAULT']['save_testing_LSTM']
            pd.DataFrame({'Actual': Actual, 'Predicted': Predicted}, index = index).to_csv(save_path + "/" + database + "_test" + str(i+1) + ".csv")

            #Predicted values of Jump Model.
            Predicted = jmpMdlPred_plot[-5:].reshape(1,-1)[0]
            #Save tested prediction result.
            save_path = config['DEFAULT']['save_testing_JmpMdl']
            pd.DataFrame({'Actual': Actual, 'Predicted': Predicted}, index = index).to_csv(save_path + "/" + database + "_test" + str(i+1) + ".csv")
            #Predicted values of ARIMA + Sentiment Analysis Model.
            Predicted = mdl_pred_plot[-5:].reshape(1,-1)[0]
            #Save tested prediction result.
            save_path = config['DEFAULT']['save_testing_ARIMASenAnyl']
            pd.DataFrame({'Actual': Actual, 'Predicted': Predicted}, index = index).to_csv(save_path + "/" + database + "_test" + str(i+1) + ".csv")

            #Start forecasting in the next horizon.
            nearestStartDate = sql_operation.last_n_day(database, startDate, frcst_h * (numOfTest - i - 1))
            nearestEndDate = sql_operation.last_n_day(database, endDate, frcst_h * (numOfTest - i - 1))
            #Change date to string format
            strNeareststartDate = str(nearestStartDate).split()[0] 
            #Change date to string format
            strNearestEndDate = str(nearestEndDate).split()[0] 

        show_msg.show_msg('Stage 2: Forecast with best model')
        #Choose best model to forecast.
        strStartDate = str(startDate).split()[0]
        strEndDate = str(endDate).split()[0]
        time_series = sql_operation.extract_all_numeric_data(database, strStartDate, strEndDate)
        actual_prices.append(sql_operation.extract_data_num(database, strEndDate, frcst_h))
        #Jump Model has minimum squared error of forecasting.
        if (SqErrJmpMdl == min(SqErrJmpMdl, SqErrARIMA, SqErrlstm)): 
            print('Forecast with Jump Model...')
            Jump_Model.matlab_simul(time_series = time_series, forecast_horizon = frcst_h, startStockPrice = actual_prices[len(actual_prices) - 1].iloc[0, 0], simul_Num = simul_Num, estimate_ii = -1, actual_prices = actual_prices, database_name = database, prev_prediction_error = jmpMdlErr, prev_prediction = jmpMdlPred, load = 1)
        elif (SqErrARIMA == min(SqErrJmpMdl, SqErrARIMA, SqErrlstm)):
            print('Forecast with ARIMA Model...')
            ARIMA.arima_mdl(time_series = time_series, frcst_h = frcst_h, estimate = 'predict', actual_prices = actual_prices[len(actual_prices) - 1], database_name = database, estimate_ii = -1, prev_prediction_error = mdl_pred_error, prev_prediction = mdl_pred_info, load = 1)
        #SqErrlstm == min(SqErrJmpMdl, SqErrARIMA, SqErrlstm)
        else: 
            print('Forecast with LSTM Model...')
            LSTM.LSTM(time_series = time_series, frcst_hz = frcst_h, actual_prices = actual_prices[len(actual_prices) - 1], database_name = database, estimate_ii = -1, prev_prediction_error = lstmErr, prev_prediction = lstmPred, load = 1)

    show_msg.show_msg('Final Stage: End of Financial Data Forecaster.')
if __name__ == "__main__":
    main()
