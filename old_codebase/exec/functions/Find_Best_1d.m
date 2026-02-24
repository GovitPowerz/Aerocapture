function [xbit,cout,nnet,row,col] = Find_Best_1d(xbit,cout,NbrSim,Input,PS)

for k = 1:PS.GA.nsubpop
    for i = 1:PS.GA.npop/PS.GA.nsubpop
        cout(k,i) = ComputeCost_1d(xbit(:,i,k),NbrSim,Input,PS);
    end
end
[tmp1,indic]=sort(cout,1);
[tmp2,col]=min(tmp1(1,:));
row = indic(1,col);
[cout_tmp,y] = ComputeCost_1d(xbit(:,col,row),NbrSim,Input,PS);
DisplayPerfo_1d(y,PS,Input);

% Improvement of the best network
for i = 1:2
    [xbit_mod,cout_mod,gain] = Improve_Chrom_1d(xbit(:,col,row),NbrSim,Input,PS,1);
    xbit(:,col,row) = xbit_mod;
    disp(['Improvement gain (%) and new cost :   ' num2str(gain) '    ' num2str(cout_mod)]);
end
[cout_tmp,y] = ComputeCost_1d(xbit(:,col,row),NbrSim,Input,PS);
DisplayPerfo_1d(y,PS,Input);

% Converting into real coefficients
nnet = sum(reshape(xbit(:,col,row),PS.GA.nbit,PS.NS.ncoef)'.*PS.GA.conv_bd,2)+PS.GA.Pmin;
