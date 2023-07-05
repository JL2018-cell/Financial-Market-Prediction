from math import sqrt
import pandas as pd
import numpy as np
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dropout
import tensorflow.keras as keras
import tensorflow as tf
import sys
import sql_operation #Local file.

#Split data in rolling windows.
#Suppose input data = [1,2,3,4] and window = 2.
#Then, output = [[1,2],[2,3],[3,4]]
def prep_data_in_rolling_window(dataset, target, start, end, window, horizon):
	X = []
	y = []
	start = start + window
	if end is None:
		end = len(dataset) - horizon
	for i in range(start, end):
		indices = range(i-window, i)
		X.append(dataset[indices])
		indicey = range(i+1, i+1+horizon)
		y.append(target[indicey])
	return np.array(X), np.array(y) 

# split a dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# fit a model
def model_fit(train, config, X_scaler, Y_scaler):
	# unpack config
	frcst_hz, hist_window, n_buffer, n_input, n_nodes, n_epochs, n_batch, n_diff = config
	# prepare data
	for i in range(n_diff):
		np.diff(train, axis = 0)
	# separate inputs and outputs and scale the data.
	X_data = X_scaler.fit_transform(train[:, :-1])
	Y_data = Y_scaler.fit_transform(train[:, -1].reshape(-1,1))

	# transform series into supervised format
	TRAIN_SPLIT = int(X_data.shape[0] * 0.8)
	x_train, y_train = prep_data_in_rolling_window(dataset = X_data, target = Y_data, start = 0, end = TRAIN_SPLIT, window = hist_window, horizon = frcst_hz)
	x_vali, y_vali = prep_data_in_rolling_window(X_data, Y_data, TRAIN_SPLIT, None, hist_window, frcst_hz) 
	train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)) #Create a source dataset from your input data.
	train_data = train_data.cache().shuffle(n_buffer).batch(n_batch).repeat() #Take first number = buffer_size data into buffer. Then, take number = n_batch elements. Once an element is taken out, then next element from data is replaced until all elements are used.
	val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali)) #Repeats this dataset so each original value is seen count times.
	val_data = val_data.batch(n_batch).repeat() 
	n_features = 5

	# define model
	lstm_model = Sequential([ #Sequential: model class. creating deep learning models fast and easy. The sequential API allows you to create models layer-by-layer for most problems. A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
		Bidirectional(LSTM(n_nodes, return_sequences=True), #n_nodes: number of neurons; set return_sequences to true since we will add more layers to the model.
			input_shape=(hist_window, n_features)), #number of time steps.
		Dense(n_nodes, activation='tanh'),
		Bidirectional(LSTM(150)), #Bidirectional: a sequence processing model that consists of two LSTMs: one taking the input in a forward direction, and the other in a backwards direction.
		Dense(n_nodes, activation='tanh'),
		Dense(n_nodes, activation='tanh'), #make the model more robust. Fully-connected layer. It's actually the layer where each neuron is connected to all of the neurons from the next layer.
		Dropout(0.2), #Dropout layer is added to avoid over-fitting, value = 0.2 means 20% of the layers will be dropped.
		Dense(units = frcst_hz),]) #frcst_hz = 5
	lstm_model.compile(optimizer='adam', loss='mse')

	model_path = 'Bidirectional_LSTM_Multivariate.h5'
	early_stopings = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min') #Stop training when a monitored metric has stopped improving.
	checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0) #Callback to save the Keras model or model weights at some frequency.
	callbacks=[early_stopings,checkpoint] 
	# fit model
	lstm_model.fit(train_data, epochs=n_epochs, steps_per_epoch=2, validation_data=val_data, validation_steps=50, verbose=1, callbacks=callbacks)
	return lstm_model

# forecast with the fit model
def model_predict(model, history, config, X_scaler, Y_scaler):
	# unpack config
	frcst_hz, hist_window, n_buffer, n_input, n_nodes, n_epochs, n_batch, n_diff = config
	# prepare data
	correction = 0.0
	if n_diff > 0:
		history = np.diff(history, n = n_diff, axis = 0)
	data_val = X_scaler.fit_transform(history)
	val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])
	return X_scaler.inverse_transform(model.predict(val_rescaled))

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	frcst_hz, hist_window, n_buffer, n_input, n_nodes, n_epochs, n_batch, n_diff = cfg
	predictions = list()
	actual = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# fit model
	X_scaler = MinMaxScaler()
	Y_scaler = MinMaxScaler()
	model = model_fit(train, cfg, X_scaler, Y_scaler)
	# seed history with training dataset
	# history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, train[-i - 1 - hist_window - n_diff:-i - 1, :-1], cfg, X_scaler, Y_scaler)
		# store forecast in list of predictions
		predictions.append(yhat)
		actual.append(data[-test.shape[1] - n_test + i:-test.shape[1] - n_test + i + frcst_hz, -1])
		# add actual observation to history for the next loop
		# history.append(test[i])
	# estimate prediction error
	error = measure_rmse(actual, [p[0] for p in predictions])
	print(' > %.3f' % error)
	return error

# score a model, return None on failure
def repeat_evaluate(data, config, n_test, n_repeats=10):
	# convert config to a key
	key = str(config)
	# fit and evaluate the model n times
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	# summarize score
	result = mean(scores)
	print('> Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test):
	# evaluate different configurations
	scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list]
	# sort configs by error in ascending order.
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a list of configs to try
def model_configs(sample_num):
	# define scope of configs
	frcst_hz = [5] #forecast horizon
	hist_window = [5, 10, 15, 20, 25, 30, 35, 40] #Historical window
	buffer_size = [sample_num//2, sample_num] #buffer size
	n_input = [5, 10, 15, 20, 25, 30, 35, 40] #number of input of the model
	n_nodes = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700] #number of nodes.
	n_epochs = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75] #number of epochs.
	n_batch = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75] #number of batchs.
	n_diff = [0,1,2] #Number of differencing the data.
	# create configs
	configs = list()
	for a in frcst_hz:
		for b in hist_window:
			for c in buffer_size:
				for i in n_input:
					for j in n_nodes:
						for k in n_epochs:
							for l in n_batch:
								for m in n_diff:
									cfg = [a, b, c, i, j, k, l, m]
									configs.append(cfg)
	print('Total configs: %d' % len(configs))
	return configs

# define dataset
series = sql_operation.extract_all_numeric_data('alibaba_9988', '2020-01-01', '2021-11-05')
series = pd.concat([series.iloc[:, :3], series.iloc[:, 4:6], series.iloc[:, 3:4]], axis = 1) #rearrange columns of dataset
data = series.values
# data split
n_test = 12
# set up configurations: forecast horizon, historical window, buffer size, input size, nodes number, epochs number, batch number, number of differencing
cfg_list = model_configs(data.shape[0])
# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
# list configs and corresponding errors
# [frcst_hz, hist_window, n_buffer, n_input, n_nodes, n_epochs, n_batch, n_diff]: Error
for cfg, error in scores:
	print(cfg, error)
