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
mfuel = 3.6;
coef_opt = 0.1;
adim_gov = [3 8];

num_sim = 10;
dvtmp = 5;
dptmp = 10;
dmtmp = m0/100;
p = [2*dptmp*rand(1,num_sim)-dptmp+100;...
    2*dvtmp*rand(1,num_sim)-dvtmp-20;...
    2*dmtmp*rand(1,num_sim)-dmtmp+m0;...
    mfuel*ones(1,num_sim)];
n_gov = size(p,2);
t = 0;
y = [p(1,:)';p(2,:)';p(3,:)'];
mass_ini = p(3,:)';
mass_fuel = p(4,:)';
dydt = y;
acc_mem = [];
y_mem = [];
t_mem = [];
while (max(abs(dydt)) > 0)
    ground = (y(1:n_gov) > 0);
    burnout = (mass_ini-y(2*n_gov+1:end) < mass_fuel);
    acc = sim(net,[y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:end)'])';
    a_gov = acc.*propmx./y(2*n_gov+1:end);
    dydt = [y(n_gov+1:2*n_gov); (a_gov.*burnout-g-1/2*rho*sref*cd_gov./y(2*n_gov+1:end).*y(n_gov+1:2*n_gov).*abs(y(n_gov+1:2*n_gov)));-y(2*n_gov+1:end).*a_gov/g0/Isp.*burnout].*[ground;ground;ground].*[burnout;burnout;burnout];
    y = y+tguid*dydt;
    t_mem = [t_mem t+tguid];
    acc_mem = [acc_mem a_gov];
    y_mem = [y_mem y];
end

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

disp(' ');
disp('Propagation :');
disp(['erreurs position (moy,std) : ' num2str(mean(max(y(1:n_gov),0)-hf)) ' ' num2str(std(max(y(1:n_gov),0)-hf))]);
disp(['erreurs vitesse (moy,std) : ' num2str(mean(y(n_gov+1:2*n_gov)-vf)) ' ' num2str(std(y(n_gov+1:2*n_gov)-vf))]);
disp(['consommation (moy,std) : ' num2str(mean(min(mass_ini-y(2*n_gov+1:3*n_gov),mfuel))) ' ' num2str(std(min(mass_ini-y(2*n_gov+1:3*n_gov),mfuel)))]);
