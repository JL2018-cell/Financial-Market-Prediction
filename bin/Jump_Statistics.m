%Called by Detect_Jumps.py
%Caclulate Poisson rate and Exponential distribution parameter from a historical time series.
function estm = jmp_stat(arr)
    arr_jmp = ischange(arr) %Indicate the position of jump.
    %dimension of matrix. m = number of rows, n = number of columns.
    [m, n] = size(arr)
    %Parameter of Poisson Distribution = \lambda = average value of samples.
    Poi_rate = sum(arr_jmp) / n
    %If there is no jump, then, it is meaningless to consider mangnitude of jump.
    if (sum(arr(arr_jmp)) == 0)
        Expn_rate = 0
    %If there is jump, then, 
    %calculate parameter of exponential distribution.
    else
        Expn_rate = sum(arr_jmp) / sum(arr(arr_jmp))
    end
estm = [Poi_rate Expn_rate]
