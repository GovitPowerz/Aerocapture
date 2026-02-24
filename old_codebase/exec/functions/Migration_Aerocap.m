function [xbit,cout,PS] = Migration_Aerocap(xbit,cout,ngen,PS)

if (round(ngen/PS.GA.migr)*PS.GA.migr == ngen)
    for l = 1:PS.GA.nsubpop-1
        xbit(:,end,l) = xbit(:,1,l+1);
        cout(l,end) = cout(l+1,1);
    end
    xbit(:,end,PS.GA.nsubpop) = xbit(:,1,1);
    cout(PS.GA.nsubpop,end) = cout(1,1);
    
    for k = 1:PS.GA.nsubpop
        [xbit_mod,cout_mod,gain] = Improve_Chrom_Aerocap(xbit(:,1,k),PS,0);
        xbit(:,end-1,k) = xbit_mod;
        cout(k,end-1) = cout_mod;
        disp(['Improvement gain (%) and new cost :   ' num2str(gain) '    ' num2str(cout_mod)]);
%        for i = 1:PS.GA.npop/PS.GA.nsubpop
%            pop_real(:,i,k) = sum(reshape(xbit(:,i,k),PS.GA.nbit,PS.NS.ncoef)'.*PS.GA.conv_bd,2)+PS.GA.Pmin;
%        end
    end
    
%    if (max(max(max(pop_real))) > PS.GA.boundcoef*PS.GA.Pmax)
%        PS.GA.Pmax = 1/PS.GA.boundcoef*PS.GA.Pmax;
%        PS.GA.conv_bd = 2.^repmat(PS.GA.nbit-1:-1:0,PS.NS.ncoef,1)/...
%                       (2^PS.GA.nbit-1)*(PS.GA.Pmax-PS.GA.Pmin);
%        disp(PS.GA.Pmax);
%        for k = 1:PS.GA.nsubpop
%            for i = 1:PS.GA.npop/PS.GA.nsubpop
%                xbit(:,i,k) = str2num(reshape(dec2bin(round((pop_real(:,i,k)-PS.GA.Pmin)/(PS.GA.Pmax-PS.GA.Pmin)*(2^PS.GA.nbit-1)))',PS.NS.ncoef*PS.GA.nbit,1));
%            end
%        end
%    end
%    if (min(min(min(pop_real))) < PS.GA.boundcoef*PS.GA.Pmin)
%        PS.GA.Pmin = 1/PS.GA.boundcoef*PS.GA.Pmin;
%        disp(PS.GA.Pmin);
%        PS.GA.conv_bd = 2.^repmat(PS.GA.nbit-1:-1:0,PS.NS.ncoef,1)/...
%                       (2^PS.GA.nbit-1)*(PS.GA.Pmax-PS.GA.Pmin);
%        for k = 1:PS.GA.nsubpop
%            for i = 1:PS.GA.npop/PS.GA.nsubpop
%                xbit(:,i,k) = str2num(reshape(dec2bin(round((pop_real(:,i,k)-PS.GA.Pmin)/(PS.GA.Pmax-PS.GA.Pmin)*(2^PS.GA.nbit-1)))',PS.NS.ncoef*PS.GA.nbit,1));
%            end
%        end
%    end
end

