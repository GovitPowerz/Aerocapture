clear all;

addpath('nncontrol/');
addpath('nndemos/');
addpath('nnet/');
addpath('nnobsolete/');
addpath('nnutils/');
warning off;

g0 = 9.80665;
Isp = 228;
sref = 0.8;
cd_gov = 1.5;
rho = 1.71e-2;
propmx = 1200;
tguid = 0.1;
g = 3.718;
m0 = 160;
vf = -5.0;
hf = 0.0;
mfuel = 19;
alt_cut = 20;

load net_1d_24_long;

num_sim = 50;
dvtmp = 5;
dptmp = 50;
dmtmp = m0/100;
p = [2*dptmp*rand(1,num_sim)-dptmp+2000;...
    2*dvtmp*rand(1,num_sim)-dvtmp-70;...
    2*dmtmp*rand(1,num_sim)-dmtmp+m0;...
    mfuel*ones(1,num_sim)];
n_gov = size(p,2);

% Neural Network
t = 0;
y = [p(1,:)';p(2,:)';p(3,:)'];
mass_ini = p(3,:)';
mass_fuel = p(4,:)';
dydt = y;
acc_mem = [];
y_mem = [];
t_mem = [];
t = 0;
while (max(abs(dydt)) > 0)
    ground = (y(1:n_gov) > 0);
    burnout = (mass_ini-y(2*n_gov+1:end) < mass_fuel);
    zero_acc = 1-((y(n_gov+1:2*n_gov) > vf).*(y(1:n_gov) < alt_cut));
    acc = sim(net,[y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:end)'])';
    a_gov = g+(acc.*propmx./y(2*n_gov+1:end)-g).*zero_acc;
    dydt = [y(n_gov+1:2*n_gov); (a_gov.*burnout-g-1/2*rho*sref*cd_gov./y(2*n_gov+1:end).*y(n_gov+1:2*n_gov).*abs(y(n_gov+1:2*n_gov)));-y(2*n_gov+1:end).*a_gov/g0/Isp.*burnout].*[ground;ground;ground].*[burnout;burnout;burnout];
    y = y+tguid*dydt;
    t = t+tguid;
    t_mem = [t_mem t];
    acc_mem = [acc_mem a_gov];
    y_mem = [y_mem y];
end
disp(' ');
disp('Propagation :');
disp(['erreurs position (moy,std) : ' num2str(mean(max(y(1:n_gov),0)-hf)) ' ' num2str(std(max(y(1:n_gov),0)-hf))]);
disp(['erreurs vitesse (moy,std) : ' num2str(mean(y(n_gov+1:2*n_gov)-vf)) ' ' num2str(std(y(n_gov+1:2*n_gov)-vf))]);
disp(['consommation (moy,std) : ' num2str(mean(min(mass_ini-y(2*n_gov+1:3*n_gov),mfuel))) ' ' num2str(std(min(mass_ini-y(2*n_gov+1:3*n_gov),mfuel)))]);

if (num_sim == 1)
    % Optimal command
    acc_mem_opt = [];
    y_mem_opt = [];
    t_mem_opt = [];
    a_gov = 0;
    init = 1;
    tgo = 30;
    mem = zeros(3,4);
    y = [p(1,:)';p(2,:)';p(3,:)'];
    mass_ini = p(3,:)';
    mass_fuel = p(4,:)';
    dydt = y;
    t = 0;
    while (max(abs(dydt)) > 0)
        ground = (y(1:n_gov) > 0);
        burnout = (mass_ini-y(2*n_gov+1:end) < mass_fuel);
        zero_acc = 1-((y(n_gov+1:2*n_gov) > vf).*(y(1:n_gov) < alt_cut));
        [comopt,tgo,mem,init] = OPTIMAL_COMMAND([0.00001;0.00001;y(1)],[0.00001;0.00001;y(2)],a_gov,tgo,mem,init);
        a_gov = g+(min(max(comopt(3),0),propmx./y(2*n_gov+1:end))-g).*zero_acc;
        dydt = [y(n_gov+1:2*n_gov); (a_gov.*burnout-g-1/2*rho*sref*cd_gov./y(2*n_gov+1:end).*y(n_gov+1:2*n_gov).*abs(y(n_gov+1:2*n_gov)));-y(2*n_gov+1:end).*a_gov/g0/Isp.*burnout].*[ground;ground;ground].*[burnout;burnout;burnout];
        y = y+tguid*dydt;
        t = t+tguid;
        t_mem_opt = [t_mem_opt t];
        acc_mem_opt = [acc_mem_opt a_gov];
        y_mem_opt = [y_mem_opt y];
    end
    disp(' ');
    disp('Propagation :');
    disp(['erreurs position (moy,std) : ' num2str(mean(max(y(1:n_gov),0)-hf)) ' ' num2str(std(max(y(1:n_gov),0)-hf))]);
    disp(['erreurs vitesse (moy,std) : ' num2str(mean(y(n_gov+1:2*n_gov)-vf)) ' ' num2str(std(y(n_gov+1:2*n_gov)-vf))]);
    disp(['consommation (moy,std) : ' num2str(mean(min(mass_ini-y(2*n_gov+1:3*n_gov),mfuel))) ' ' num2str(std(min(mass_ini-y(2*n_gov+1:3*n_gov),mfuel)))]);
end

if (num_sim > 1)
    figure;
    subplot(2,2,1);
    plot(t_mem,acc_mem,'+');
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
else
    figure;
    subplot(2,2,1);
    plot(t_mem,acc_mem,'+',t_mem_opt,acc_mem_opt,'+');
    grid on;
    xlabel('time (s)');
    ylabel('acceleration (m/s2)');
    legend('Neural','Optimal',0);
    subplot(2,2,2);
    plot(t_mem,y_mem(2*n_gov+1:end,:),'+',t_mem_opt,y_mem_opt(2*n_gov+1:end,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('mass (kg)');
    legend('Neural','Optimal',0);
    subplot(2,2,3);
    plot(t_mem,y_mem(1:n_gov,:),'+',t_mem_opt,y_mem_opt(1:n_gov,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('position (m)');
    legend('Neural','Optimal',0);
    subplot(2,2,4);
    plot(t_mem,y_mem(n_gov+1:2*n_gov,:),'+',t_mem_opt,y_mem_opt(n_gov+1:2*n_gov,:),'+');
    grid on;
    xlabel('time (s)');
    ylabel('velocity (m/s)');
    legend('Neural','Optimal',0);
end
