function [net] = init_net_1d(net,input,ndiv,nplot);


indic_gov = 0;
save indic_gov indic_gov;
net.trainParam.show = 1;
net.trainParam.epochs = 200;
net.trainParam.goal = 1e-14;
net.trainParam.mu_max = 1e12;
% Chargement des paramètres physiques
load prop_param;

n_gov = size(input,2);
t = 0;
% Initialisation du vecteur d'état [pos(z),vit(z),mass]
y = [input(1,:)';input(2,:)';input(3,:)'];
mass_ini = input(3,:)';
mass_fuel = input(4,:)';
dydt = y;
count = 0;
% Propagation et stockage
input = [];
output = [];
acc_mem = [];
y_mem = [];
t_mem = [];
while (max(abs(dydt)) > 0)
    % Condition d'arrêt sur impact avec le sol
    ground = (y(1:n_gov) > 0);
    % Condition d'arrêt sur consommation ergols
    burnout = (mass_ini-y(2*n_gov+1:end) < mass_fuel);
    % Condition d'arrêt sur critère vitesse verticale (pour éviter
    % d'atteindre des vitesses positives)
    stop_gov = (y(n_gov+1:2*n_gov) < vf/2);
    % On fige la commande lorsque l'altitude est faible et la vitesse
    % verticale a atteint la vitesse désirée
    zero_acc = 1-((y(n_gov+1:2*n_gov) > vf).*(y(1:n_gov) < alt_cut));
    zero_acc2 = 1-(y(1:n_gov) < alt_cut);
    % Commande donnée par le réseau 1d pour [posz,vitz,mass]
    az_gov = 0.0;
    az_gov = g+(az_gov-g).*zero_acc;
    % Saturation de la commande par l'accélération disponible
    az_gov = max(min(az_gov,propmx./y(4*n_gov+1:end)),0);
    % Stockage des entrées du futur réseau 2d et de la commande
    % correspondante
    if ((floor(count/ndiv) == count/ndiv)&&(ndiv~=0))
        input = [input [y(1:n_gov)';...
            y(n_gov+1:2*n_gov)';...
            y(2*n_gov+1:end)']];
        output = [output [az_gov'./propmx.*y(4*n_gov+1:end)']];
    end
    % Calcul de la norme de la vitesse pour les forces aéros
    vit_gov = sqrt(y(n_gov+1:2*n_gov).^2);
    % Calcul de la dérivé du vecteur d'état
    dydt = [y(n_gov+1:2*n_gov);...
        (az_gov.*burnout-g...
        -1/2*rho*sref*cd_gov./y(4*n_gov+1:end).*vit_gov.*y(n_gov+1:2*n_gov));...
        -y(4*n_gov+1:end).*sqrt(az_gov.^2)/g0/Isp.*burnout]...
        .*[ground;ground;ground;ground;ground]...
        .*[burnout;burnout;burnout;burnout;burnout]...
        .*[stop_gov;stop_gov;stop_gov;stop_gov;stop_gov];
    % Calcul de nouveau vecteur d'état
    y = y+tguid*dydt;
    t = t+tguid;
    count = count + 1;
    t_mem = [t_mem t];
    acc_mem = [acc_mem az_gov];
    y_mem = [y_mem y];
end

% Visualisation des n premières trajectoires
if (nplot > 0)
    n = min(nplot,n_gov);
    figure;
    subplot(2,2,1);
    plot(t_mem,acc_mem(1:n,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('acceleration (m/s2)');
    subplot(2,2,2);
    plot(t_mem,y_mem(2*n_gov+1:end,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('mass (kg)');
    subplot(2,2,3);
    plot(t_mem,y_mem(1:n_gov,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('position (m)');
    subplot(2,2,4);
    plot(t_mem,y_mem(n_gov+1:2*n_gov,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('velocity (m/s)');
end

disp(' ');
disp('Propagation :');
disp(['erreurs position (moy,std,min,max) : ' num2str(mean(max(y(1:n_gov),0)-0.0)) ' ' num2str(std(max(y(1:n_gov),0)-0.0)) ' ' num2str(min(max(y(1:n_gov),0)-0.0)) ' ' num2str(max(max(y(1:n_gov),0)-0.0))]);
disp(['erreurs vitesse (moy,std,min,max) : ' num2str(mean(y(n_gov+1:2*n_gov)-vf)) ' ' num2str(std(y(n_gov+1:2*n_gov)-vf)) ' ' num2str(min(y(n_gov+1:2*n_gov)-vf)) ' ' num2str(max(y(n_gov+1:2*n_gov)-vf))]);
disp(['consommation (moy,std,min,max) : ' num2str(mean(mfuel-max(mass_fuel-(mass_ini-y(2*n_gov+1:3*n_gov)),0))) ' ' num2str(std(mfuel-max(mass_fuel-(mass_ini-y(2*n_gov+1:3*n_gov)),0))) ' ' num2str(min(mfuel-max(mass_fuel-(mass_ini-y(2*n_gov+1:3*n_gov)),0))) ' ' num2str(max(mfuel-max(mass_fuel-(mass_ini-y(2*n_gov+1:3*n_gov)),0)))]);
disp(' ');
% Entrainement du réseau 1d
disp(['Nombre de cas : ' num2str(size(input,2))]);
pause;
[net,tr] = train(net,input,output);
