import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px #to plot the time series plot
from sklearn import metrics #for evaluating models.
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import tensorflow as tf 
import tensorflow.keras as keras
import configparser

#Preprocesses the data suitable for forecasting.
def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
     X = []
     y = []
     start = start + window
     if end is None:
         end = len(dataset) - horizon
     for i in range(start, end):
         X.append(dataset[range(i-window, i)])
         y.append(target[range(i+1, i+1+horizon)])
     return np.array(X), np.array(y) 

def LSTM(time_series, frcst_hz, actual_prices, database_name, estimate_ii, prev_prediction_error, prev_prediction, load):
    print('LSTM is proecessing data...')
    config = configparser.ConfigParser()
    config.read('../conf/setting.ini')
    #If main.py tells LSTM is the best model of forecast, then, 
    #load model saved in ../temp/ and forecast.
    if (estimate_ii < 0):
        if load == 1: #Load LSTM model
            hist_window = frcst_hz * 2
            X_scaler = MinMaxScaler()
            Y_scaler = MinMaxScaler()
            #Prepare data (independent variables)
            X_data = X_scaler.fit_transform(time_series.iloc[:, 1:len(time_series.columns)])
            #Prepare data (dependent variable)
            Y_data = Y_scaler.fit_transform(time_series['Close'].to_numpy().reshape(-1,1))
            #Load model.
            reconstructed_model = keras.models.load_model(config['DEFAULT']['LSTM_model_read_path'])
            data_val = X_scaler.fit_transform(time_series.iloc[:, 1:len(time_series.columns)].tail(hist_window))
            val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])
            pred = reconstructed_model.predict(val_rescaled)
            #Undo the scaling according to feature_range.
            pred_Inverse = Y_scaler.inverse_transform(pred)
            index = ['Day' + str(i) for i in range(1, len(pred_Inverse[0]) + 1)]
            pred_result_pd = pd.DataFrame(pred_Inverse[0], index = index, columns = ['Forecasted value'])
            pred_result_pd.index.name = "Predicted by LSTM Model"
            pred_result_pd.to_csv('../result/forecast_%s.csv' % database_name.split('_')[0])
    #Testing and estimating model.
    else:
        #Scale data such that they are easy to train models.
        X_scaler = MinMaxScaler()
        Y_scaler = MinMaxScaler()
        X_data = X_scaler.fit_transform(time_series.iloc[:, 1:len(time_series.columns)])
        Y_data = Y_scaler.fit_transform(time_series['Close'].to_numpy().reshape(-1,1))
        
        #Number of training data in each trial
        hist_window = frcst_hz * 2
        #Number of data classified as train data.
        TRAIN_SPLIT = int(X_data.shape[0]*0.8)
        #Convert Training Data to Right Shape
        x_train, y_train = custom_ts_multi_data_prep(X_data, Y_data, 0, TRAIN_SPLIT, hist_window, frcst_hz)
        x_vali, y_vali = custom_ts_multi_data_prep(X_data, Y_data, TRAIN_SPLIT, None, hist_window, frcst_hz) 
    
        #Prepare the training data and validation data using the TensorFlow data function, which faster and efficient way to feed data for training.
        batch_size = 10
        buffer_size = int(X_data.shape[0]*0.8)
        #Create a source dataset from your input data.
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        #Take first number = buffer_size data into buffer. 
        #Then, take number = batch_size elements. 
        #Once an element is taken out, then,
        #next element from data is replaced until all elements are used.
        train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat() 
        #Repeats this dataset so each original value is seen count times.
        val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
        val_data = val_data.batch(batch_size).repeat() 

        #Define LSTM Model.
        #Sequential: model class.
        #a plain stack of layers where each layer has exactly 
        #one input tensor and one output tensor.
        lstm_model = keras.models.Sequential([ 
          #200: number of neurons; 
          #set return_sequences to true since more layers will be added.
          keras.layers.Bidirectional(keras.layers.LSTM(600, return_sequences=True),
                                       input_shape=x_train.shape[-2:]), #number of time steps.
             keras.layers.Dense(410, activation='tanh'),
             #Bidirectional: a sequence processing model that consists of two LSTMs: 
             #one taking the input in a forward direction, 
             #and the other in a backwards direction.
             keras.layers.Bidirectional(keras.layers.LSTM(150)), 
             keras.layers.Dense(510, activation='tanh'),
             keras.layers.Dense(420, activation='tanh'), 
             #Dropout layer is added to avoid over-fitting, 
             #value = 0.2 means 20% of the layers will be dropped.
             keras.layers.Dropout(0.1),
             keras.layers.Dense(units=frcst_hz),])
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.summary() 

        #A string to tell where to save progress of model training.
        model_path = 'Bidirectional_LSTM_Multivariate.h5' 

        #Quantity to be monitored = 'val_loss', 
        #min_delta = Minimum change in the monitored quantity to qualify as an improvement, 
        #patience = Number of epochs with no improvement after which training will be stopped,
        #verbose = verbosity mode, 
        #mode = training will stop when the quantity monitored has stopped decreasing.
        early_stopings = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min') #Stop training when a monitored metric has stopped improving.
        checkpoint = keras.callbacks.ModelCheckpoint(config['DEFAULT']['LSTM_model_run_path'], monitor='val_loss', save_best_only=True, mode='min', verbose=0) #Callback to save the Keras model or model weights at some frequency.
        callbacks = [early_stopings,checkpoint] 
        
        history = lstm_model.fit(train_data,epochs=1,steps_per_epoch=2,validation_data=val_data,validation_steps=50,verbose=1,callbacks=callbacks)
        #Epoch: indicates the number of passes of the entire training dataset the machine learning algorithm has completed.
        #Epoch ~ iteration
        #One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters.
        #The batch size is a number of samples processed before the model is updated.
        #The number of epochs is the number of complete passes through the training dataset.

        #Prepare forecasting.
        #Prepare Input data
        data_val = X_scaler.fit_transform(time_series.iloc[:, 1:len(time_series.columns)].tail(hist_window)) 
        #Reshape input data
        val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])
        #Forecast
        pred = lstm_model.predict(val_rescaled)
        #Undo the scaling according to feature_range. This is prediction result.
        pred_Inverse = Y_scaler.inverse_transform(pred)

        #Calculate squared prediction error.
        pred_sq_error = np.linalg.norm(np.linalg.norm(actual_prices.iloc[:,0] - pred_Inverse[0]))**2
        #First time of estimation: save model and 
        #return squared prediction error and forecast result.
        if (estimate_ii == 0): 
            lstm_model.save(config['DEFAULT']['LSTM_model_read_path'])
            return (pred_sq_error, pred_Inverse[0])
        #Not first time of estimation: 
        #if this prediction error < previous prediction error, then ,
        #save model, and 
        #return squared prediction error and forecast result. 
        #Otherwise, return previous prediction error and forecast result.
        else: #estimate_ii > 0 
            if (pred_sq_error < prev_prediction_error):
                lstm_model.save(config['DEFAULT']['LSTM_model_read_path'])
                return (pred_sq_error, pred_Inverse[0])
            else:
                return (prev_prediction_error, prev_prediction)


