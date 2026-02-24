function test_visu_net_2d(net,num_sim)

% Initialisation des parametres physiques
init_param_2d;

input = [2*dptmp(1)*rand(1,num_sim)-dptmp(1)+posnom(1);...
    2*dptmp(2)*rand(1,num_sim)-dptmp(2)+posnom(2);...
    2*dvtmp(1)*rand(1,num_sim)-dvtmp(1)+vitnom(1);...
    2*dvtmp(2)*rand(1,num_sim)-dvtmp(2)+vitnom(2);...
    2*dmtmp*rand(1,num_sim)-dmtmp+masnom;...
    mfuel*ones(1,num_sim)];

n_gov = size(input,2);

% Neural Network
% Propagation des conditions initiales avec un reseau 2d
t = 0;
% Initialisation du vecteur d'etat [pos(x,z),vit(x,z),mass]
y = [input(1,:)';input(2,:)';input(3,:)';input(4,:)';input(5,:)'];
mass_ini = input(5,:)';
mass_fuel = input(6,:)';
dydt = y;
% Propagation et stockage
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
    t_mem = [t_mem t];
    acc_mem = [acc_mem [ax_gov;az_gov;sqrt((ax_gov).^2+(az_gov).^2)]];
    y_mem = [y_mem y];
end

disp(' ');
disp('Propagation :');
disp(['erreurs position hori (moy,std,min,max) : ' num2str(mean(y(1:n_gov))) ' ' num2str(std(y(1:n_gov))) ' ' num2str(min(y(1:n_gov))) ' ' num2str(max(y(1:n_gov)))]);
disp(['erreurs position vert (moy,std,min,max) : ' num2str(mean(max(y(n_gov+1:2*n_gov),0))) ' ' num2str(std(max(y(n_gov+1:2*n_gov),0))) ' ' num2str(min(max(y(n_gov+1:2*n_gov),0))) ' ' num2str(max(max(y(n_gov+1:2*n_gov),0)))]);
disp(['erreurs vitesse hori (moy,std,min,max) : ' num2str(mean(y(2*n_gov+1:3*n_gov))) ' ' num2str(std(y(2*n_gov+1:3*n_gov))) ' ' num2str(min(y(2*n_gov+1:3*n_gov))) ' ' num2str(max(y(2*n_gov+1:3*n_gov)))]);
disp(['erreurs vitesse vert (moy,std,min,max) : ' num2str(mean(y(3*n_gov+1:4*n_gov)-vf)) ' ' num2str(std(y(3*n_gov+1:4*n_gov)-vf)) ' ' num2str(min(y(3*n_gov+1:4*n_gov)-vf)) ' ' num2str(max(y(3*n_gov+1:4*n_gov)-vf))]);
disp(['consommation (moy,std,min,max) : ' num2str(mean(mfuel-max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0))) ' ' num2str(std(mfuel-max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0))) ' ' num2str(min(mfuel-max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0))) ' ' num2str(max(mfuel-max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0)))]);

if (num_sim == 1)
    % Optimal command
    acc_mem_opt = [];
    y_mem_opt = [];
    t_mem_opt = [];
    ax_gov = 0;
    ay_gov = 0;
    init = 1;
    tgo = 30;
    mem = zeros(3,4);
    y = [input(1,:)';input(2,:)';input(3,:)';input(4,:)';input(5,:)'];
    mass_ini = input(5,:)';
    mass_fuel = input(6,:)';
    dydt = y;
    t = 0;
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
        % Commande donnee par la commande optimale
        [comopt,tgo,mem,init] = OPTIMAL_COMMAND([y(1);0.00001;y(2)],[y(3);0.00001;y(4)],sqrt((ax_gov).^2+(az_gov).^2),tgo,mem,init);
        ax_gov = comopt(1);
        az_gov = max(comopt(3),0);
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
        t_mem_opt = [t_mem_opt t];
        acc_mem_opt = [acc_mem_opt [ax_gov;az_gov;sqrt((ax_gov).^2+(az_gov).^2)]];
        y_mem_opt = [y_mem_opt y];
    end
    disp(' ');
    disp('Propagation :');
    disp(['erreurs position hori (moy,std,min,max) : ' num2str(mean(y(1:n_gov))) ' ' num2str(std(y(1:n_gov))) ' ' num2str(min(y(1:n_gov))) ' ' num2str(max(y(1:n_gov)))]);
    disp(['erreurs position vert (moy,std,min,max) : ' num2str(mean(max(y(n_gov+1:2*n_gov),0))) ' ' num2str(std(max(y(n_gov+1:2*n_gov),0))) ' ' num2str(min(max(y(n_gov+1:2*n_gov),0))) ' ' num2str(max(max(y(n_gov+1:2*n_gov),0)))]);
    disp(['erreurs vitesse hori (moy,std,min,max) : ' num2str(mean(y(2*n_gov+1:3*n_gov))) ' ' num2str(std(y(2*n_gov+1:3*n_gov))) ' ' num2str(min(y(2*n_gov+1:3*n_gov))) ' ' num2str(max(y(2*n_gov+1:3*n_gov)))]);
    disp(['erreurs vitesse vert (moy,std,min,max) : ' num2str(mean(y(3*n_gov+1:4*n_gov)-vf)) ' ' num2str(std(y(3*n_gov+1:4*n_gov)-vf)) ' ' num2str(min(y(3*n_gov+1:4*n_gov)-vf)) ' ' num2str(max(y(3*n_gov+1:4*n_gov)-vf))]);
    disp(['consommation (moy,std,min,max) : ' num2str(mean(mfuel-max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0))) ' ' num2str(std(mfuel-max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0))) ' ' num2str(min(mfuel-max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0))) ' ' num2str(max(mfuel-max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0)))]);
end

if (num_sim > 1)
    figure;
    subplot(2,2,1);
    plot(t_mem,acc_mem(1:n_gov,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('acceleration along x (m/s2)');
    subplot(2,2,2);
    plot(t_mem,acc_mem(n_gov+1:2*n_gov,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('acceleration along z (m/s2)');
    subplot(2,2,3);
    plot(t_mem,acc_mem(2*n_gov+1:3*n_gov,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('acceleration (m/s2)');
    subplot(2,2,4);
    plot(t_mem,y_mem(4*n_gov+1:end,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('mass (kg)');

    figure;
    subplot(2,2,1);
    plot(t_mem,y_mem(n_gov+1:2*n_gov,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('position along z (m)');
    subplot(2,2,2);
    plot(t_mem,y_mem(1:n_gov,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('position along x (m)');
    subplot(2,2,3);
    plot(t_mem,y_mem(3*n_gov+1:4*n_gov,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('velocity along z (m/s)');
    subplot(2,2,4);
    plot(t_mem,y_mem(2*n_gov+1:3*n_gov,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('velocity along x (m/s)');
else
    figure;
    subplot(2,2,1);
    plot(t_mem,acc_mem(1:n_gov,:),'+',t_mem_opt,acc_mem_opt(1:n_gov,:),'o');
    grid on;
    xlabel('time (s)');
    ylabel('acceleration along x (m/s2)');
    legend('Neural','Optimal',0);
    subplot(2,2,2);
    plot(t_mem,acc_mem(n_gov+1:2*n_gov,:),'+',t_mem_opt,acc_mem_opt(n_gov+1:2*n_gov,:),'o');
    grid on;
    xlabel('time (s)');
    ylabel('acceleration along z (m/s2)');
    legend('Neural','Optimal',0);
    subplot(2,2,3);
    plot(t_mem,acc_mem(2*n_gov+1:3*n_gov,:),'+',t_mem_opt,acc_mem_opt(2*n_gov+1:3*n_gov,:),'o');
    grid on;
    xlabel('time (s)');
    ylabel('acceleration (m/s2)');
    legend('Neural','Optimal',0);
    subplot(2,2,4);
    plot(t_mem,y_mem(4*n_gov+1:end,:),'+',t_mem_opt,y_mem_opt(4*n_gov+1:end,:),'o');
    grid on;
    xlabel('time (s)');
    ylabel('mass (kg)');
    legend('Neural','Optimal',0);

    figure;
    subplot(2,2,1);
    plot(t_mem,y_mem(n_gov+1:2*n_gov,:),'+',t_mem_opt,y_mem_opt(n_gov+1:2*n_gov,:),'o');
    grid on;
    xlabel('time (s)');
    ylabel('position along z (m)');
    legend('Neural','Optimal',0);
    subplot(2,2,2);
    plot(t_mem,y_mem(1:n_gov,:),'+',t_mem_opt,y_mem_opt(1:n_gov,:),'o');
    grid on;
    xlabel('time (s)');
    ylabel('position along x (m)');
    legend('Neural','Optimal',0);
    subplot(2,2,3);
    plot(t_mem,y_mem(3*n_gov+1:4*n_gov,:),'+',t_mem_opt,y_mem_opt(3*n_gov+1:4*n_gov,:),'o');
    grid on;
    xlabel('time (s)');
    ylabel('velocity along z (m/s)');
    legend('Neural','Optimal',0);
    subplot(2,2,4);
    plot(t_mem,y_mem(2*n_gov+1:3*n_gov,:),'+',t_mem_opt,y_mem_opt(2*n_gov+1:3*n_gov,:),'o');
    grid on;
    xlabel('time (s)');
    ylabel('velocity along x (m/s)');
    legend('Neural','Optimal',0);
end
pause;
