function [xbit,cout] = Initial_Population_Aerocap(PS)

for k = 1:PS.GA.nsubpop
    npop_init = 5*PS.GA.npop/PS.GA.nsubpop;
    xbit_init = round(rand(PS.GA.nbit*PS.NS.ncoef,npop_init));
    %xbit_init = round(rand(PS.GA.nbit*PS.NS.ncoef+PS.NS.ncoef/2,npop_init));
    cout_init = zeros(1,npop_init);
    for i = 1:npop_init
        cout_init(i) = ComputeCost_Aerocap(xbit_init(:,i),PS,0);
        disp(['Initial Chromosome num. ' num2str(i) ',  Cost : ' num2str(cout_init(i))]);
    end
    best = [cout_init' (1:length(cout_init))'];
    indic = sortrows(best);
    cout(k,:) = cout_init(indic(1:PS.GA.npop/PS.GA.nsubpop,2));
    xbit(:,:,k) = xbit_init(:,indic(1:PS.GA.npop/PS.GA.nsubpop,2));
end
for k = 1:PS.GA.nsubpop
    [xbit_mod,cout_mod,gain] = Improve_Chrom_Aerocap(xbit(:,1,k),PS,0);
    xbit(:,end,k) = xbit_mod;
    cout(k,end) = cout_mod;
    disp(['Improvement gain (%) and new cost :   ' num2str(gain) '    ' num2str(cout_mod)]);
end
