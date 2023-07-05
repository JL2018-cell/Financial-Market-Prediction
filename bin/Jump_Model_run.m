%jmp: Number of jumps occured
%P_lmb: Poisson Process parameter
%drift, StdD: Parameters of Brownian Motion
%E_lmb: Exponential Distribution parameter
%numS: Number of Steps
%simNum: Number of simulations
%St_ini: Initial price of stock

%function e = expFunc(N)
%e = 2.718281828^N

function simul_result = simul6(jmp, P_lmb, drift, StdD, E_lmb, numS, simNum, St_ini)
%Switch random seed every time.
rng shuffle

%Initial price
St = St_ini

%Record price movement.
St_chg = double(zeros(1,0))
%Record average price movement.
St_chg_Avg = double(zeros(1, numS))


%Simulate many times.
for i = 1:simNum
    %Each simulation is divided into many steps.
    for t = 1:numS
        %Calculate aggregate magnitude of jumps
        tmp1 = poissrnd(P_lmb) %Number of jumps
        tmp2 = double(ones(1, tmp1)) %Prepare a vector with length = Number of jumps
        tmp3 = exprnd(tmp2 * E_lmb) %each element of the vector = magnitude of jump ~ Exponential distribution.
        prd = prod(tmp3) %Multiply all elements in vectors together.
        %Geometric Brownian Motion * A series of Jumps
        St = St * double(2.718281828^((drift - (StdD*StdD)/2) * 1 + StdD * normrnd(0,1))) * prd
        St_chg = [St_chg double(St)] %Record price movement.
    end
    St_chg_Avg = double(St_chg) + double(St_chg_Avg)
    %Prepare for next simulation. Set beginning price = initial price.
    St = St_ini
    %Prepare for next simulation. Clear price movement in this simulation.
    St_chg = double(zeros(1,0))

end
%return average price movement.
simul_result = St_chg_Avg / double(simNum)
end