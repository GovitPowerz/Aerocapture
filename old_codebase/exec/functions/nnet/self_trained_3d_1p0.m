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
adim_gov = [0.01 14 3 34 3 34];

load net_1d_6_new6;
num_sim = 10000;
p = [200*rand(1,num_sim);...
    200*rand(1,num_sim)-100;...
    200*rand(1,num_sim)-100;...
    80*rand(1,num_sim)-40;...
    80*rand(1,num_sim)-40;...
    80*rand(1,num_sim)-40;...
    20*rand(1,num_sim)+150];
res_1d = sim(net,[p(1,:);p(4,:);p(7,:)]);
t = [res_1d;0.5*ones(size(res_1d));0.5*ones(size(res_1d))];

net = newff([0 200;-100 100;-100 100;-40 40;-40 40;-40 40;150 170],[12,3],{'tansig','logsig'},'trainlm');

indic_gov = 0;
save indic_gov indic_gov;
net.trainParam.show = 1;
net.trainParam.epochs = 200;
net.trainParam.goal = 1e-14;
net.trainParam.mu_max = 1e12;
[net,tr] = train(net,p,t);


% Training
indic_gov = 3;
save indic_gov indic_gov;
train_record = [];
net.trainParam.show = 1;
net.trainParam.epochs = 25;
net.trainParam.goal = 1e-14;
net.trainParam.mu_max = 1e12;
coef = 1;
num_sim = 500;
for j = 1:4
    dvtmp = coef*[0.5;0.25;0.25];
    dptmp = coef*[1.0;1.0;1.0];
    dmtmp = m0/100;
    for i = 1:20
        p = [2*dptmp(1)*rand(1,num_sim)-dptmp(1)+100;...
            2*dptmp(2)*rand(1,num_sim)-dptmp(2);...
            2*dptmp(3)*rand(1,num_sim)-dptmp(3);...
            2*dvtmp(1)*rand(1,num_sim)-dvtmp(1)-20;...
            2*dvtmp(2)*rand(1,num_sim)-dvtmp(2);...
            2*dvtmp(3)*rand(1,num_sim)-dvtmp(3);...
            2*dmtmp*rand(1,num_sim)-dmtmp+m0;...
            mfuel*ones(1,num_sim)];
        n_gov = size(p,2);
        mf_rest = p(8,:);
        save prop_param m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov mf_rest;

        t = ones(3,size(p,2));
        [net,tr] = train(net,p(1:7,:),t);
        train_record = [train_record tr.perf];
    end
    num_sim = 1*num_sim;
    coef = coef*(10)^(1/3);
    net.trainParam.epochs = 2*net.trainParam.epochs;
end

figure;
semilogy(train_record);


