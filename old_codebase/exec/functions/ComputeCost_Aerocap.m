function [cout,nnet_out,CondFin] = ComputeCost_Aerocap(xbit,PS,mode)

if (mode == 0)
    % Conversion binaire -> decimal
    nnet = sum(reshape(xbit(1:PS.GA.nbit*PS.NS.ncoef),PS.GA.nbit,PS.NS.ncoef)'.*...
        PS.GA.conv_bd,2)+PS.GA.Pmin;

    %nnet_out = sign(xbit(PS.GA.nbit*PS.NS.ncoef+1:end)-0.5).*PS.NS.nnet.*(1+PS.GA.Var*(nnet(PS.NS.ncoef/2+1:end)+1)/2.*nnet(1:end-PS.NS.ncoef/2));
    nnet_out = sign(xbit(PS.GA.nbit*PS.NS.ncoef+1:end)).*PS.NS.nnet.*(1+PS.GA.Var*(nnet(PS.NS.ncoef/3+1:2*PS.NS.ncoef/3)+1)/2.*nnet(1:PS.NS.ncoef/3));
%     nnet_out = PS.NS.nnet+PS.GA.Var*(nnet(PS.NS.ncoef/2+1:end)+1)/2./nnet(1:end-PS.NS.ncoef/2);
    %plot(PS.GA.Var*(nnet(PS.NS.ncoef/2+1:end)+1)/2./nnet(1:end-PS.NS.ncoef/2),'+')
    %pause
else
    nnet_out = PS.NS.nnet;
end

fid = fopen('../donnees/nn_param.temp','wt');

fprintf(fid,'  \n');
fprintf(fid,'   Caracteristiques neural network\n');
fprintf(fid,'  \n');
fprintf(fid,['           ' num2str(PS.NS.ninput) '   ninput\n']);
fprintf(fid,['           ' num2str(PS.NS.nhid) '   nhid\n']);
fprintf(fid,['           ' num2str(PS.NS.noutput) '   noutput\n']);
for i = 1:length(nnet_out)
	fprintf(fid,'%40.30f\n',nnet_out(i));
end

fclose(fid);

eval(['!aerocap_nn < ' PS.SC.initfile ' > ecran.temp']);

load('../sorties/final.temp');
CondFin = abs([final(:,29) final(:,39) final(:,40) final(:,41)]);

%CondFin(:,4) = zeros(size(CondFin(:,4)));
%cond = (CondFin(:,1) > 1e10);
%err = cond.*1e20./CondFin(:,1)+~cond.*(final(:,43));
%cond2 = (CondFin(:,1) < 450);
%err = cond.*1e30./CondFin(:,1)+~cond.*(cond2.*1e16./CondFin(:,1)+~cond2.*(0*abs(CondFin(:,2))+abs(CondFin(:,3))));

crash = ((CondFin(:,2) > 1e20)|(CondFin(:,3) > 1e20)|(CondFin(:,4) > 1e20));
apo = (CondFin(:,3) > 40);
peri = (CondFin(:,2)-113 > 40);
incli = (CondFin(:,4) > 40);
err = crash.*1e30./CondFin(:,1)+~crash.*...
    (apo.*(1e18*CondFin(:,3)+1e12*CondFin(:,1)+1e6*CondFin(:,4))+~apo.*...
    (peri.*(CondFin(:,3)+1e12*CondFin(:,1)+1e6*CondFin(:,4))+~peri.*...
    (incli.*(CondFin(:,3)+CondFin(:,1)+1e6*CondFin(:,4))+~incli.*...
    (CondFin(:,3)+CondFin(:,1)+CondFin(:,4)))));

cout = sqrt(sum(sum(err.^2))/numel(err));
