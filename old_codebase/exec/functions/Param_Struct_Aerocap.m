function PS = Param_Struct_Aerocap

% Structure Name :
% PS -> Parameters Structure
% SC -> Simulation Condition
% BP -> BackPropagation parameters
% NS -> Network Size
% GA -> Genetic Algorithm parameters
% VP -> Various Parameters

indic_old = 1;
if (indic_old == 1)
    load('save_net/optim_net_Aerocap_18-Jul-2007')
    old_nnet = PS.NS.nnet;
    clear PS
end

% Genetic algo param
PS.SC.initfile = 'aerocap.in_msr_aller_64_nn';
PS.NS.ninput = 7;
PS.NS.nhid = 24;
PS.NS.noutput = 2;
PS.NS.ncoef = (PS.NS.ninput+PS.NS.noutput)*PS.NS.nhid+PS.NS.nhid+PS.NS.noutput;
if (indic_old == 0)
    PS.NS.nnet = 1e-1*(2*rand(PS.NS.ncoef,1)-1);
else
    PS.NS.nnet = old_nnet;
end
PS.NS.mincout = 1e30;
PS.NS.ncoef = 2*PS.NS.ncoef;
PS.GA.nbit = 32;
PS.GA.Pmax = 1;
PS.GA.Pmin = -1;
PS.GA.Var = 1e-1;
PS.GA.npop = 20;
PS.GA.nsubpop = 1;
PS.GA.boundcoef = 1.0;
PS.GA.migr = 10;
PS.GA.ngen = 100;
PS.GA.mut_coef = 0.01;
PS.GA.conv_bd = 2.^repmat(PS.GA.nbit-1:-1:0,PS.NS.ncoef,1)/...
    (2^PS.GA.nbit-1)*(PS.GA.Pmax-PS.GA.Pmin);

fid = fopen('../donnees/param_algo','wt');

fprintf(fid,'      integer  ninput,nhid,noutput,ncoef,ncoeftot\n');
fprintf(fid,'c        \n');
fprintf(fid,'      double precision  inphid,biashd,hidout,biasout\n');
fprintf(fid,'c      \n');
fprintf(fid,'c     Genetic algo param\n');
fprintf(fid,'c\n');
fprintf(fid,['      parameter (ninput = ' num2str(PS.NS.ninput) ')\n']);
fprintf(fid,['      parameter (nhid = ' num2str(PS.NS.nhid) ')\n']);
fprintf(fid,['      parameter (noutput = ' num2str(PS.NS.noutput) ')\n']);
fprintf(fid,'      parameter (ncoef = (ninput+noutput)*nhid+nhid+noutput)\n');
fprintf(fid,'      parameter (ncoeftot = ncoef)\n');
fprintf(fid,'c\n');
fprintf(fid,'      common / paramnnd / inphid(nhid,ninput),biashd(nhid),\n');
fprintf(fid,'     +                    hidout(noutput,nhid),biasout(noutput)\n');
fprintf(fid,'c\n');

fclose(fid);
!make clean
!make
% ?????
PS.VP.visu = 0;
PS.VP.indic_old = indic_old;
