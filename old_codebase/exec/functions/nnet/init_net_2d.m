function [net] = init_net_2d(net,net_1d,input,ndiv,nplot);

indic_gov = 0;
save indic_gov indic_gov;
net.trainParam.show = 1;
net.trainParam.epochs = 200;
net.trainParam.goal = 1e-14;
net.trainParam.mu_max = 1e12;
% Chargement des parametres physiques
load prop_param;

% Propagation des conditions initiales avec un reseau 1d pour la poussee
% verticale et un Gravity Turn pour le lateral
n_gov = size(input,2);
t = 0;
% Initialisation du vecteur d'etat [pos(x,z),vit(x,z),mass]
y = [input(1,:)';input(2,:)';input(3,:)';input(4,:)';input(5,:)'];
mass_ini = input(5,:)';
mass_fuel = input(6,:)';
dydt = y;
count = 0;
% Propagation et stockage
input = [];
output = [];
acc_mem = [];
y_mem = [];
t_mem = [];
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
    % Commande donnee par le reseau 1d pour [posz,vitz,mass]
    az_gov = propmx./y(4*n_gov+1:end).*sim(net_1d,[y(n_gov+1:2*n_gov)';y(3*n_gov+1:4*n_gov)';y(4*n_gov+1:end)'])';
    az_gov = g+(az_gov-g).*zero_acc;
    % Commande laterale par Gravity Turn ax = Vx/Vz*az
    ax_gov = -y(2*n_gov+1:3*n_gov)./abs(y(3*n_gov+1:4*n_gov)./az_gov);
    ax_gov = -y(1:n_gov)./abs(y(n_gov+1:2*n_gov)./az_gov)-y(2*n_gov+1:3*n_gov)./abs(y(3*n_gov+1:4*n_gov)./az_gov);
    % Saturation de la commande par l'acceleration disponible
    acc = max(sqrt((ax_gov).^2+(az_gov).^2),1e-10);
    a_gov = max(min(acc,propmx./y(4*n_gov+1:end)),1e-10);
    ax_gov = ax_gov.*a_gov./acc;
    az_gov = az_gov.*a_gov./acc;
    % Stockage des entrees du futur reseau 2d et de la commande
    % correspondante
    if ((floor(count/ndiv) == count/ndiv)&&(ndiv~=0))
        input = [input [y(1:n_gov)';...
            y(n_gov+1:2*n_gov)';...
            y(2*n_gov+1:3*n_gov)';...
            y(3*n_gov+1:4*n_gov)';...
            y(4*n_gov+1:end)']];
        output = [output [ax_gov'./propmx.*y(4*n_gov+1:end)'/2+0.5;...
            az_gov'./propmx.*y(4*n_gov+1:end)']];
    end
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
    t_mem = [t_mem t];
    acc_mem = [acc_mem [ax_gov;az_gov;sqrt((ax_gov).^2+(az_gov).^2)]];
    y_mem = [y_mem y];
end

% Visualisation des n premieres trajectoires
if (nplot > 0)
    n = min(nplot,n_gov);
    figure;
    subplot(2,2,1);
    plot(t_mem,acc_mem(1:n,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('acceleration along x (m/s2)');
    subplot(2,2,2);
    plot(t_mem,acc_mem(n_gov+1:n_gov+n,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('acceleration along y (m/s2)');
    subplot(2,2,3);
    plot(t_mem,acc_mem(2*n_gov+1:2*n_gov+n,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('acceleration (m/s2)');
    subplot(2,2,4);
    plot(t_mem,y_mem(4*n_gov+1:4*n_gov+n,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('mass (kg)');

    figure;
    subplot(2,2,1);
    plot(t_mem,y_mem(n_gov+1:n_gov+n,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('position along z (m)');
    subplot(2,2,2);
    plot(t_mem,y_mem(1:n,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('position along x (m)');
    subplot(2,2,3);
    plot(t_mem,y_mem(3*n_gov+1:3*n_gov+n,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('velocity along z (m/s)');
    subplot(2,2,4);
    plot(t_mem,y_mem(2*n_gov+1:2*n_gov+n,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('velocity along x (m/s)');
end

disp(' ');
disp('Propagation :');
disp(['erreurs position hori (moy,std,min,max) : ' num2str(mean(y(1:n_gov))) ' ' num2str(std(y(1:n_gov))) ' ' num2str(min(y(1:n_gov))) ' ' num2str(max(y(1:n_gov)))]);
disp(['erreurs position vert (moy,std,min,max) : ' num2str(mean(max(y(n_gov+1:2*n_gov),0))) ' ' num2str(std(max(y(n_gov+1:2*n_gov),0))) ' ' num2str(min(max(y(n_gov+1:2*n_gov),0))) ' ' num2str(max(max(y(n_gov+1:2*n_gov),0)))]);
disp(['erreurs vitesse hori (moy,std,min,max) : ' num2str(mean(y(2*n_gov+1:3*n_gov))) ' ' num2str(std(y(2*n_gov+1:3*n_gov))) ' ' num2str(min(y(2*n_gov+1:3*n_gov))) ' ' num2str(max(y(2*n_gov+1:3*n_gov)))]);
disp(['erreurs vitesse vert (moy,std,min,max) : ' num2str(mean(y(3*n_gov+1:4*n_gov)-vf)) ' ' num2str(std(y(3*n_gov+1:4*n_gov)-vf)) ' ' num2str(min(y(3*n_gov+1:4*n_gov)-vf)) ' ' num2str(max(y(3*n_gov+1:4*n_gov)-vf))]);
disp(['consommation (moy,std,min,max) : ' num2str(mean(mfuel-max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0))) ' ' num2str(std(mfuel-max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0))) ' ' num2str(min(mfuel-max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0))) ' ' num2str(max(mfuel-max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0)))]);
disp(' ');
% Entrainement du reseau 2d
disp(['Nombre de cas : ' num2str(size(input,2))]);
pause;
[net,tr] = train(net,input,output);
