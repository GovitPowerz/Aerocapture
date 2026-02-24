clear all;

addpath('nncontrol/');
addpath('nndemos/');
addpath('nnet/');
addpath('nnobsolete/');
addpath('nnutils/');
warning off;

% Initialisation des parametres physiques
init_param_2d;

disp('Initialisation du reseau 2d...');
disp(' ');
load net_1d_24_long;
net_1d = net;
net = newff([-500 500;0 2000;-50 50;-120 40;150 170],[48,2],{'tansig','logsig'},'trainlm');

num_sim = 1000;
input = [2*dptmp(1)*rand(1,num_sim)-dptmp(1)+posnom(1);...
    2*dptmp(2)*rand(1,num_sim)-dptmp(2)+posnom(2);...
    2*dvtmp(1)*rand(1,num_sim)-dvtmp(1)+vitnom(1);...
    2*dvtmp(2)*rand(1,num_sim)-dvtmp(2)+vitnom(2);...
    2*dmtmp*rand(1,num_sim)-dmtmp+masnom;...
    mfuel*ones(1,num_sim)];

ndiv = 20;
nplot = 10;
[net] = init_net_2d(net,net_1d,input,ndiv,nplot);

disp('Visualisation des perfs du reseau 2d apres init...');
disp(' ');
num_sim = 10;
test_visu_net_2d(net,num_sim)

%save net_2d_init_48 net;
%load net_2d_init_48;

disp(' ');
disp('Entrainement du reseau 2d...');
disp(' ');
% Training
indic_gov = 2;
save indic_gov indic_gov;
train_record = [];
net.trainParam.show = 1;
net.trainParam.epochs = 25;
net.trainParam.goal = 1e-14;
net.trainParam.mu_max = 1e12;
coef = 0.1;
dvtmp = coef*dvtmp;
dptmp = coef*dptmp;
coef = 1;
num_sim = 200;
for j = 1:4
    dvtmp = coef*dvtmp;
    dptmp = coef*dptmp;
    ndiv = 0;
    for i = 1:10
        input = [2*dptmp(1)*rand(1,num_sim)-dptmp(1)+posnom(1);...
            2*dptmp(2)*rand(1,num_sim)-dptmp(2)+posnom(2);...
            2*dvtmp(1)*rand(1,num_sim)-dvtmp(1)+vitnom(1);...
            2*dvtmp(2)*rand(1,num_sim)-dvtmp(2)+vitnom(2);...
            2*dmtmp*rand(1,num_sim)-dmtmp+masnom;...
            mfuel*ones(1,num_sim)];
        n_gov = size(input,2);
        % Propagation des conditions initiales avec un reseau 2d
        t = 0;
        % Initialisation du vecteur d'etat [pos(x,z),vit(x,z),mass]
        y = [input(1,:)';input(2,:)';input(3,:)';input(4,:)';input(5,:)'];
        mass_ini = input(5,:)';
        mass_fuel = input(6,:)';
        dydt = y;
        ndiv = ndiv+0;
        count = floor((ndiv-1)*rand(1));
        while (max(abs(dydt)) > 0)
            % Condition d'arret sur impact avec le sol
            ground = (y(n_gov+1:2*n_gov) > 0);
            % Condition d'arret sur consommation ergols
            burnout = (mass_ini-y(4*n_gov+1:end) < mass_fuel);
            % Condition d'arret sur critere vitesse verticale (pour eviter
            % d'atteindre des vitesses positives)
            stop_gov = (y(3*n_gov+1:4*n_gov) < vf/2);
            % On fige la commande lorsque l'altitude est faible et la vitesse
            % verticale a atteint la vitesse desiree
            zero_acc = 1-((y(3*n_gov+1:4*n_gov) > vf).*(y(n_gov+1:2*n_gov) < alt_cut));
            zero_acc2 = 1-(y(n_gov+1:2*n_gov) < alt_cut);
            % Commande donnee par le reseau 2d
            tmp = sim(net,[y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:3*n_gov)';...
                y(3*n_gov+1:4*n_gov)';y(4*n_gov+1:end)'])';
            ax_gov = propmx./y(4*n_gov+1:end).*(2*(tmp(:,1)-0.5));
            az_gov = propmx./y(4*n_gov+1:end).*tmp(:,2);
            az_gov = g+(az_gov-g).*zero_acc;
            % Saturation de la commande par l'acceleration disponible
            acc = max(sqrt((ax_gov).^2+(az_gov).^2),1e-10);
            a_gov = max(min(acc,propmx./y(4*n_gov+1:end)),1e-10);
            ax_gov = ax_gov.*a_gov./acc;
            az_gov = az_gov.*a_gov./acc;
            % Calcul de la norme de la vitesse pour les forces aeros
            vit_gov = sqrt(y(2*n_gov+1:3*n_gov).^2+y(3*n_gov+1:4*n_gov).^2);
            % Calcul de la derive du vecteur d'etat
            dydt = [y(2*n_gov+1:4*n_gov);...
                (ax_gov.*burnout...
                -1/2*rho*sref*cd_gov./y(4*n_gov+1:end).*vit_gov.*y(2*n_gov+1:3*n_gov));...
                (az_gov.*burnout-g...
                -1/2*rho*sref*cd_gov./y(4*n_gov+1:end).*vit_gov.*y(3*n_gov+1:4*n_gov));...
                -y(4*n_gov+1:end).*sqrt(ax_gov.^2+az_gov.^2)/g0/Isp.*burnout]...
                .*[ground;ground;ground;ground;ground]...
                .*[burnout;burnout;burnout;burnout;burnout]...
                .*[stop_gov;stop_gov;stop_gov;stop_gov;stop_gov];
            % Calcul de nouveau vecteur d'etat
            y = y+tguid*dydt;
            t = t+tguid;
            count = count + 1;
            % Stockage des entrees du futur reseau 2d et de la commande
            if ((floor(count/ndiv) == count/ndiv)&&(count~=0)&&(ndiv~=0))
                input = [input [y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:3*n_gov)';y(3*n_gov+1:4*n_gov)';y(4*n_gov+1:end)';max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0)']];
            end
        end
        mf_rest = input(6,:);
        save prop_param m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov mf_rest alt_cut;

        t = ones(2,size(input,2));
        [net,tr] = train(net,input(1:5,:),t);
        train_record = [train_record tr.perf];
    end
    num_sim = num_sim+100;
    coef = coef*(10)^(1/3);
    net.trainParam.epochs = 2*net.trainParam.epochs;
end

figure;
semilogy(train_record);


