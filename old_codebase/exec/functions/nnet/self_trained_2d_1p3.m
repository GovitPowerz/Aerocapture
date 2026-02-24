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
m0 = 152;
vf = -5.0;
hf = 0.0;
mfuel = 19;
coef_opt = 0.1;
adim_gov = [0.01 8 5 90];
alt_cut = 20;

load net_1d_6_long;
net_action = net;
save net_1d_action net_action;

net_lat = newff([0 2000;-500 500;-120 40;-50 50;150 170],[24,1],{'tansig','logsig'},'trainlm');

indic_gov = 0;
save indic_gov indic_gov;
net_lat.trainParam.show = 1;
net_lat.trainParam.epochs = 50;
net_lat.trainParam.goal = 1e-14;
net_lat.trainParam.mu_max = 1e12;
num_sim = 5000;
p = [2000*rand(1,num_sim);...
    1000*rand(1,num_sim)-500;...
    160*rand(1,num_sim)-120;...
    100*rand(1,num_sim)-50;...
    20*rand(1,num_sim)+150];
t = 0.5+0.01*rand(1,size(p,2))-0.005;
[net_lat,tr] = train(net_lat,p,t);
% save net_2d_init_48 net;
%load net_2d_init_48;

% Training
indic_gov = 5;
save indic_gov indic_gov;
train_record = [];
net_lat.trainParam.show = 1;
net_lat.trainParam.epochs = 25;
net_lat.trainParam.goal = 1e-14;
net_lat.trainParam.mu_max = 1e12;
coef = 0.1;
num_sim = 100;
for j = 1:4
    dvtmp = coef*[5;0.5];
    dptmp = coef*[50;10.0];
    dmtmp = m0/100;
    ndiv = 20;
    for i = 1:20
        p = [2*dptmp(1)*rand(1,num_sim)-dptmp(1)+200;...
            2*dptmp(2)*rand(1,num_sim)-dptmp(2);...
            2*dvtmp(1)*rand(1,num_sim)-dvtmp(1)-10;...
            2*dvtmp(2)*rand(1,num_sim)-dvtmp(2);...
            2*dmtmp*rand(1,num_sim)-dmtmp+m0;...
            mfuel*ones(1,num_sim)];
        n_gov = size(p,2);
        t = 0;
        y = [p(1,:)';p(2,:)';p(3,:)';p(4,:)';p(5,:)'];
        mass_ini = p(5,:)';
        mass_fuel = p(6,:)';
        dydt = y;
        ndiv = ndiv+0;
        count = floor((ndiv-1)*rand(1));
        while (max(abs(dydt)) > 0)
            ground = (y(1:n_gov) > 0);
            burnout = (mass_ini-y(4*n_gov+1:end) < mass_fuel);
            zero_acc = 1-((y(2*n_gov+1:3*n_gov) > vf).*(y(1:n_gov) < alt_cut));
            zero_acc2 = 1-(y(1:n_gov) < alt_cut);
            stop_gov = (y(2*n_gov+1:3*n_gov) < vf/2);
            tmp = sim(net_lat,[y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:3*n_gov)';...
                y(3*n_gov+1:4*n_gov)';y(4*n_gov+1:end)'])';
            ax_gov = 2*(0.5*tmp-0.25);
            ay_gov = sim(net_action,[y(1:n_gov)';y(2*n_gov+1:3*n_gov)';y(4*n_gov+1:end)'])';
            acc = max(sqrt((ax_gov).^2+(ay_gov).^2),1e-8);
            a_gov = max(min(acc,1),1e-8);
            ax_gov = propmx*ax_gov.*a_gov./acc./y(4*n_gov+1:end);
            ay_gov = g+(propmx*ay_gov.*a_gov./acc./y(4*n_gov+1:end)-g).*zero_acc;
            vit_gov = sqrt(y(2*n_gov+1:3*n_gov).^2+y(3*n_gov+1:4*n_gov).^2);
            dydt = [y(2*n_gov+1:3*n_gov);y(3*n_gov+1:4*n_gov);...
                (ay_gov.*burnout-g...
                -1/2*rho*sref*cd_gov./y(4*n_gov+1:end).*vit_gov.*y(2*n_gov+1:3*n_gov));...
                (ax_gov.*burnout...
                -1/2*rho*sref*cd_gov./y(4*n_gov+1:end).*vit_gov.*y(3*n_gov+1:4*n_gov));...
                -y(4*n_gov+1:end).*sqrt(ax_gov.^2+ay_gov.^2)/g0/Isp.*burnout].*[ground;ground;ground;ground;ground]...
                .*[burnout;burnout;burnout;burnout;burnout].*[stop_gov;stop_gov;stop_gov;stop_gov;stop_gov]...
                .*[zero_acc2;zero_acc2;zero_acc2;zero_acc2;zero_acc2];
            y = y+tguid*dydt;
            if ((floor(count/ndiv) == count/ndiv)&&(count~=0)&&(ndiv~=0)&&(min(y(1:n_gov) > 10) > 0)&&(min(abs(dydt)) > 0))
                p = [p [y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:3*n_gov)';y(3*n_gov+1:4*n_gov)';y(4*n_gov+1:end)';max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0)']];
            end
            t = t+tguid;
            count = count + 1;
        end
        mf_rest = p(6,:);
        save prop_param m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov mf_rest alt_cut;

        t = ones(1,size(p,2));
        [net_lat,tr] = train(net_lat,p(1:5,:),t);
        train_record = [train_record tr.perf];
    end
    num_sim = num_sim+100;
    coef = coef*(10)^(1/3);
    net.trainParam.epochs = 2*net.trainParam.epochs;
end

figure;
semilogy(train_record);


