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
mfuel = 5;
coef_opt = 0.1;
adim_gov = [0.01 14 0.1 14];

load net_1d_6_new6;
num_sim = 10000;
p = [200*rand(1,num_sim);...
    200*rand(1,num_sim)-100;...
    80*rand(1,num_sim)-40;...
    80*rand(1,num_sim)-40;...
    20*rand(1,num_sim)+150];
res_1d = sim(net,[p(1,:);p(3,:);p(5,:)]);
t = [res_1d;0.5*ones(size(res_1d))];

net = newff([0 200;-100 100;-40 40;-40 40;150 170],[14,2],{'tansig','logsig'},'trainlm');

indic_gov = 0;
save indic_gov indic_gov;
net.trainParam.show = 1;
net.trainParam.epochs = 100;
net.trainParam.goal = 1e-14;
net.trainParam.mu_max = 1e12;
[net,tr] = train(net,p,t);


% Training
indic_gov = 2;
save indic_gov indic_gov;
train_record = [];
net.trainParam.show = 1;
net.trainParam.epochs = 25;
net.trainParam.goal = 1e-14;
net.trainParam.mu_max = 1e12;
coef = 1;
ndiv = 5000000;
num_sim = 800;
for j = 1:4
    dvtmp = coef*[0.5;0.25];
    dptmp = coef*[1.0;1.0];
    dmtmp = m0/100;
    for i = 1:20
        p = [2*dptmp(1)*rand(1,num_sim)-dptmp(1)+100;...
            2*dptmp(2)*rand(1,num_sim)-dptmp(2);...
            2*dvtmp(1)*rand(1,num_sim)-dvtmp(1)-20;...
            2*dvtmp(2)*rand(1,num_sim)-dvtmp(2);...
            2*dmtmp*rand(1,num_sim)-dmtmp+m0;...
            mfuel*ones(1,num_sim)];
        n_gov = size(p,2);
        t = 0;
        y = [p(1,:)';p(2,:)';p(3,:)';p(4,:)';p(5,:)'];
        mass_ini = p(5,:)';
        mass_fuel = p(6,:)';
        dydt = y;
        ndiv = ndiv+5;
        count = floor((ndiv-1)*rand(1));
        while (max(abs(dydt)) > 0)
            ground = (y(1:n_gov) > 0);
            burnout = (mass_ini-y(4*n_gov+1:end) < mass_fuel);
            tmp = sim(net,[y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:3*n_gov)';...
                y(3*n_gov+1:4*n_gov)';y(4*n_gov+1:end)'])';
            ax_gov = 2*(tmp(:,2)-0.5);
            ay_gov = tmp(:,1);
            acc = sqrt((ax_gov).^2+(ay_gov).^2);
            a_gov = max(min(acc,1),1e-8);
            ax_gov = propmx*ax_gov.*a_gov./acc./y(4*n_gov+1:end);
            ay_gov = propmx*ay_gov.*a_gov./acc./y(4*n_gov+1:end);
            vit_gov = sqrt(y(2*n_gov+1:3*n_gov).^2+y(3*n_gov+1:4*n_gov).^2);
            dydt = [y(2*n_gov+1:3*n_gov);y(3*n_gov+1:4*n_gov);...
                (ay_gov.*burnout-g...
                -1/2*rho*sref*cd_gov./y(4*n_gov+1:end).*vit_gov.*y(2*n_gov+1:3*n_gov));...
                (ax_gov.*burnout...
                -1/2*rho*sref*cd_gov./y(4*n_gov+1:end).*vit_gov.*y(3*n_gov+1:4*n_gov));...
                -y(4*n_gov+1:end).*sqrt(ax_gov.^2+ay_gov.^2)/g0/Isp.*burnout].*[ground;ground;ground;ground;ground]...
                .*[burnout;burnout;burnout;burnout;burnout];
            y = y+tguid*dydt;
            if ((floor(count/ndiv) == count/ndiv)&&(count~=0)&&(ndiv~=0)&&(min(y(1:n_gov) > 10) > 0)&&(min(abs(dydt)) > 0))
                p = [p [y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:3*n_gov)';y(3*n_gov+1:4*n_gov)';y(4*n_gov+1:end)';max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0)']];
            end
            t = t+tguid;
            count = count + 1;
        end
        mf_rest = p(6,:);
        save prop_param m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov mf_rest;

        t = ones(2,size(p,2));
        [net,tr] = train(net,p(1:5,:),t);
        train_record = [train_record tr.perf];
    end
    ndiv = ndiv+100000;
    num_sim = 1*num_sim;
    coef = coef*(10)^(1/3);
    net.trainParam.epochs = 2*net.trainParam.epochs;
end

figure;
semilogy(train_record);


