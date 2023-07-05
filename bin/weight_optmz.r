deviation_error <- function(data, par) {
    #1 - correlation between data$x and exponential decay of data$y
    return (1 - cor(data$x * par[1]^(1:length(data$y)), data$y)^2)
    }
fndWght <- function(data){
    #initial guess = 0.5, 
    #function to minimize = deviation_error, 
    #method of optimization = "BFGS".
    optim_output <- optim(par = c(0.5), fn = deviation_error, method = "BFGS", data = data) #uses a quasiNewton method. This helps to speed up execution.
    return (optim_output)
    }
