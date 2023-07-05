library(forecast)

#df = DataFrame of time series
#test stationarity of a time series: mean-stationary and standard deviation-stationary.
test_stationarity <- function(df) {
    #Divide whole time series into 10 intervals.
    interval_num <- 10

    #Calculate number of data in each interval.
    interval_len <- length(df) %/% 10

    #Tolerate a little fluctuation in standard deviation and mean of time series is different intervals.
    sd_sd_tolerenace <- 0.1
    mean_sd_tolerance <- 0.1

    #Calculate standard deviation and mean of each interval.
    #Results are stored in array intervals_sd and intervals_mean respectively.
    intervals_sd <- sd(df[1:interval_len])
    intervals_mean <- mean(df[1:interval_len])
    for (i in 2:(interval_num - 1)) {
        cat(i, "\n")
        intervals_sd[i] <- sd(df[((i - 1) * interval_len): (i * interval_len)])
        intervals_mean[i] <- mean(df[((i - 1) * interval_len): (i * interval_len)])
    }
    intervals_sd[i+1] <- sd(df[(i * interval_len): ((i + 1) * interval_len)])
    intervals_mean[i+1] <- mean(df[(i * interval_len): ((i + 1) * interval_len)])

    #Determine if given time series is stationary. If yes, return True, else return False.
    if (sd(intervals_sd) <= sd_sd_tolerenace & sd(intervals_mean) <= mean_sd_tolerance) {
        cat("Stationary!\n")
        TRUE
    } else {
        cat("Non-stationary!\n")
        FALSE
    }
}

Forecast_r_function= function(time_series, f_h, estm_ii, actual_prices, prev_prediction_error, prev_prediction, load, load_path = "../temp/ARIMA_model.rda") {
    #If main.py tells ARIMA is the best model of forecast, 
    #then, load model saved in ../temp/ and forecast.
    if (estm_ii < 0) {
        if (load==1) {
            load(load_path) #Load model.
            print("Load success!")
            as.vector(t(as.data.frame(forecast(fit, f_h))["Point Forecast"])) #Forecast according to the model and return forecasted values.
        }
    }
    #Testing and estimating model.
    else {
        #Test stationarity of time series, the definition of this function is written above.
        if (test_stationarity(time_series) == FALSE) {
            cat("Warning! This time series is NOT stationary. So, the following ARIMA estimation may NOT be reliable.")
        }
        #Convert data to time series format in R.
        y <- ts(time_series)
        #Estimate by Autoregressive Integrated Moving Average Model. 
        #This function can make non-stationary time series into stationary one by 
        #taking appropriate number of differencing.
        #Trace = TRUE means displaying estimation precedure on screen.
        fit <- auto.arima(y, trace = TRUE)
        #This is the first time of estimation.
        if (estm_ii == 0) {
            forecast_prices = as.vector(t(as.data.frame(forecast(fit, f_h))["Point Forecast"]))
            this_prediction_error = norm(actual_prices - forecast_prices, type = "2")^2
            save(fit, file = load_path)
            c(this_prediction_error, forecast_prices)
        }
        #This is not the first time of estimation. 
        #So, we need to compare previous prediction error and this prediction error.
        else {
            forecast_prices = as.vector(t(as.data.frame(forecast(fit, f_h))["Point Forecast"]))
            this_prediction_error = norm(actual_prices - forecast_prices, type = "2")^2
            #If former error > later error, then, save this model, and 
            #return this forecast error and forecast values.
            if (this_prediction_error < prev_prediction_error) {
                save(fit, file = load_path)
                c(this_prediction_error, forecast_prices)
            }
            #If former error < later error, then, 
            #former forecast is better than this forecast.
            #So, model estimated here is discarded.
            else {
                c(prev_prediction_error, prev_prediction)
            }
        }
    }
}
