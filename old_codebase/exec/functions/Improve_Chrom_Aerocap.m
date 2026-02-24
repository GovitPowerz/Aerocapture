function [xbit_mod,cout_mod,gain] = Improve_Chrom_Aerocap(xbit,PS,mode)

if (mode == 0)
    numcoef = 1;
else
    numcoef = PS.NS.ncoef;
end

cout = ComputeCost_Aerocap(xbit,PS,0);
gain = cout;
disp(['Old cost :   ' num2str(gain)]);
tmp = sortrows([rand(PS.NS.ncoef,1) (1:PS.NS.ncoef)']);
ordre = tmp(:,2);
for i = 1:numcoef
    tmp = sortrows([rand(PS.GA.nbit,1) (1:PS.GA.nbit)']);
    ordre_bit = tmp(:,2);
    for j = 1:PS.GA.nbit
        xbit_tmp = xbit;
        xbit_tmp((ordre(i)-1)*PS.GA.nbit+ordre_bit(j)) = ~xbit_tmp((ordre(i)-1)*PS.GA.nbit+ordre_bit(j));
        cout_tmp = ComputeCost_Aerocap(xbit_tmp,PS,0);
        if (cout_tmp < cout)
            xbit = xbit_tmp;
            cout = cout_tmp;
        end
    end
end
%tmp = sortrows([rand(PS.NS.ncoef/2,1) (1:PS.NS.ncoef/2)']);
%ordre = tmp(:,2);
%for j = 1:PS.NS.ncoef/2
%    xbit_tmp = xbit;
%    xbit_tmp(PS.NS.ncoef*PS.GA.nbit+ordre(j)) = ~xbit_tmp(PS.NS.ncoef*PS.GA.nbit+ordre(j));
%    cout_tmp = ComputeCost_Aerocap(xbit_tmp,PS,0);
%    if (cout_tmp < cout)
%        xbit = xbit_tmp;
%        cout = cout_tmp;
%    end
%end
xbit_mod = xbit;
cout_mod = cout;
gain = 100*(1-cout/gain);
