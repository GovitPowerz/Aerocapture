clear all;
close all;
addpath('functions/');
rehash;
warning off;
format short g;

for ll = 1:100
PS = Param_Struct_Aerocap;

[PS.NS.mincout,tmp_disp2,CondFin_Ini] = ComputeCost_Aerocap([],PS,1);
disp(' ');
disp('        Duration    Periapsis    Apoapsis    Inclination :')
disp(CondFin_Ini)
disp(' ');
disp(['Initial cost : ' num2str(PS.NS.mincout)]);
disp(' ');

disp('Initial population');
NbrSim = 10;
[xbit,cout] = Initial_Population_Aerocap(PS);

NbrSim = 50;
cout_mem = [];
indic_new_min = 0;
for j = 1:PS.GA.ngen
    tic;
    disp(' ');
    disp(['Generation number: ' int2str(j) '/' int2str(PS.GA.ngen)]);
    % Ecriture du fichier de lancement
    fid = fopen('../exec/aerocap.in_msr_aller_64_nn','wt');
    fprintf(fid,'1                              natman          aerocapture (1) ou aerogravity assist (2)\n');
    fprintf(fid,'4                           natpla                nature Planete: Terre 3  Mars 4  Jupiter  5\n');
    fprintf(fid,'50                          nbsimu          nombre de simulations a jouer\n');
    fprintf(fid,'1                          natsim          totalite du guidage (1), capture (2) ou sortie (3) ou preprogramme (4)\n');
    fprintf(fid,'2                          natgnn          guidage ftc (1), guidage nn (2)\n');
    fprintf(fid,'0                           istats          traitement statistique uniquement (1)\n');
    fprintf(fid,'1                          itirag          lecture (0) ou creation (1) des dispersions\n');
    fprintf(fid,'1                           numsim          numero de simulation a rejouer\n');
    fprintf(fid,'0                           isauve          sauvegarde (1) des resultats de la simulation a visualiser\n');
    fprintf(fid,'1                           numvis          numero de simulation a visualiser\n');
    fprintf(fid,'0                           iecran          edition ecran (1) ou non (0) de messages intermediaires\n');
    fprintf(fid,'0.6866                           xgalea              generateur aleatoire\n');
    %fprintf(fid,[num2str(rand) '                           xgalea              generateur aleatoire\n']);
    fprintf(fid,'0                             irefer              trajectoire de reference (1) ou guidee (0)\n');
    fprintf(fid,'64.77026                            gitref              gite constante sur trajectoire de reference\n');
    fprintf(fid,'1.                              xmulti(1)       coeff. multiplicatif erreurs nav aerocapture\n');
    fprintf(fid,'1.                              xmulti(2)       coeff. multiplicatif erreurs nav interplanetaire\n');
    fprintf(fid,'1.                              xmulti(3)       coeff. multiplicatif erreurs mesure accelero\n');
    fprintf(fid,'1.                              xmulti(4)       coeff. multiplicatif erreurs modele aero\n');
    fprintf(fid,'.msr_aller                     sufmsr           caracteristiques capsule\n');
    fprintf(fid,'.msr_aller64                     sufren           caracteristiques rentree\n');
    fprintf(fid,'.msr_aller                     sufmis           caracteristiques mission\n');
    fprintf(fid,'.msr_aller64                     sufgui               caracteristiques guidage vol equilibre - phase de sortie\n');
    fprintf(fid,'.temp                     sufgnn               caracteristiques guidage neural\n');
    fprintf(fid,'.msr_aller                     sufinc           caracteristiques profil d incidence commandee\n');
    fprintf(fid,'.msr                           sufaer           caracteristiques aerodynamiques\n');
    fprintf(fid,'.mars_ASTRIUM                          sufatm           caracteristiques atmospheriques\n');
    fprintf(fid,'.msr_essai                               sufdis           caracteristiques dispersions initiales\n');
    fprintf(fid,'.nul                     sufnav           caracteristiques navigation\n');
    fprintf(fid,'.msr2                           suflot           caracteristiques dispersions-meconnaissances\n');
    fprintf(fid,'.msr_aller                           sufsuc           caracteristiques erreurs finales admissibles\n');
    fprintf(fid,'.temp   sufres             fichier de resultat\n');
    fprintf(fid,'1                                            confirmation des choix\n');
    fclose(fid);
    % Fin d'ecriture du fichier de lancement
    %disp(' ');
	%[tmp_disp1,tmp_disp2,CondFin_Best] = ComputeCost_Aerocap(xbit(:,1),PS,1);
	%disp('        Duration    Periapsis    Apoapsis    Inclination :')
    %disp(CondFin_Best)
    %disp(' ');
    %PS.NS.mincout = tmp_disp1;
    %disp(['Previous min cost : ' num2str(PS.NS.mincout)]);
    %disp(' ');
    for k = 1:PS.GA.nsubpop
        if (PS.GA.nsubpop ~=1)
            disp(['Subpopulation number: ' int2str(k) '/' int2str(PS.GA.nsubpop)]);
        end
        fitness = (max(cout(k,:))-cout(k,:))/max(cout(k,:));
        if (max(fitness) == 0)
            fitness = rand(size(fitness));
        end
        sector = fitness/sum(fitness);
        sect_cs = cumsum(sector);

        new_chrom = zeros(1,PS.GA.npop/PS.GA.nsubpop);
        for i = 1:PS.GA.npop/PS.GA.nsubpop
            wheel = rand(1,1);
            tmp = find(sect_cs >= wheel);
            new_chrom(i) = tmp(1);
        end

        % reproduction
        disp('Reproduction');
        for i = 1:PS.GA.npop/PS.GA.nsubpop/2
            reprod = round(rand(size(xbit,1),1));
            xbit_new(:,2*i-1) = reprod.*xbit(:,new_chrom(2*i-1),k)+(1-reprod).*xbit(:,new_chrom(2*i),k);
            xbit_new(:,2*i) = (1-reprod).*xbit(:,new_chrom(2*i-1),k)+reprod.*xbit(:,new_chrom(2*i),k);
        end

        % mutation
        disp('Mutation');
        n_mut = ceil(PS.GA.mut_coef*numel(xbit_new));
        pos = ceil(rand(n_mut,1)*numel(xbit_new));
        xbit_new(pos) = ~xbit_new(pos);

        %cost evaluation
        disp('Cost evaluation');
        for i = 1:PS.GA.npop/PS.GA.nsubpop
            [cout(k,i),nnet_tmp,CondFin{i}] = ComputeCost_Aerocap(xbit(:,i,k),PS,0);
	        if (cout(k,i) < PS.NS.mincout)
                PS.NS.mincout = cout(k,i);
                nnet_sub = nnet_tmp;
                indic_new_min = 1;
            end
            [cout_new(i),nnet_new_tmp,CondFin{i+PS.GA.npop/PS.GA.nsubpop}] = ComputeCost_Aerocap(xbit_new(:,i),PS,0);
	        if (cout_new(i) < PS.NS.mincout)
                PS.NS.mincout = cout_new(i);
                nnet_sub = nnet_new_tmp;
                indic_new_min = 1;
            end
        end
        xbit_tmp = [xbit(:,:,k) xbit_new];
        cout_tmp = [cout(k,:) cout_new];
        best = [cout_tmp' (1:length(cout_tmp))'];
        indic = sortrows(best);
        cout(k,:) = cout_tmp(indic(1:PS.GA.npop/PS.GA.nsubpop,2));
        xbit(:,:,k) = xbit_tmp(:,indic(1:PS.GA.npop/PS.GA.nsubpop,2));
        %[couttmp,y] = ComputeCost_1d(xbit(:,1,k),NbrSim,Input,PS);
        %DisplayPerfo_1d(y,PS,Input);
        disp(' ');
		disp('        Duration    Periapsis    Apoapsis    Inclination :')
        disp(CondFin{indic(1,2)})
        disp(' ');
        disp(['Min cost : ' num2str(min(cout(k,:)))]);
        disp(' ');
    end
    [xbit,cout,PS] = Migration_Aerocap(xbit,cout,j,PS);
    cout_mem = [cout_mem min(cout')'];
    %Evolution_Plot(cout,cout_mem,j);
	if (indic_new_min == 1)
        PS.NS.nnet = nnet_sub;
    end
    indic_new_min = 0;
	[tmp_disp1,tmp_disp2,CondFin_Best] = ComputeCost_Aerocap(xbit(:,1),PS,1);
	disp('        Duration    Periapsis    Apoapsis    Inclination :')
    disp(CondFin_Best)
    disp(' ');
    disp(['All time min cost : ' num2str(PS.NS.mincout)]);
    save(['save_net/optim_net_Aerocap_' date],'cout_mem','xbit','cout','PS');
    toc;
end
end
fid = fopen('../donnees/nn_param.temp','wt');

fprintf(fid,'  \n');
fprintf(fid,'   Caracteristiques neural network\n');
fprintf(fid,'  \n');
fprintf(fid,['           ' num2str(PS.NS.ninput) '   ninput\n']);
fprintf(fid,['           ' num2str(PS.NS.nhid) '   nhid\n']);
fprintf(fid,['           ' num2str(PS.NS.noutput) '   noutput\n']);
for i = 1:length(PS.NS.nnet)
	fprintf(fid,'%40.30f\n',PS.NS.nnet(i));
end

fclose(fid);

