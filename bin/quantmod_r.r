library('quantmod')
#name = symbol of financial product.
data_collection = function(name, startDate) {
    #Ignore warnings. This does not affect data collected.
    options("getSymbols.warning4.0"=FALSE) 

    data <- getSymbols(name, auto.assign = FALSE, src = "yahoo", from = startDate)

    #Ignore warnings. This does not affect data collected.
    options("getSymbols.warning4.0"=FALSE) 

    return (as.data.frame(data))
}
