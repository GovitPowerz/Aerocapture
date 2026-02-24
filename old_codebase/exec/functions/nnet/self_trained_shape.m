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
tguid = 0.5;
g = 3.718;
m0 = 160;
vf = -5.0;
hf = 0.0;
mfuel = 4.0;
coef_opt = 1;
adim_gov = [3 18];
save prop_param m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov;

net = newff([0 200;-40 40;150 170;0 mfuel],[52,20],{'tansig','logsig'},'trainlm');

%init
net.trainParam.show = 1;
net.trainParam.epochs = 100;
net.trainParam.goal = 1e-9;
net.trainParam.mu_max = 1e12;

% num_sim = 2000;
% p = [200*rand(1,num_sim);80*rand(1,num_sim)-40;2*m0/100*rand(1,num_sim)-m0/100+m0;mfuel*ones(1,num_sim)];
% t = 0.2*ones(1,size(p,2));
% indic_gov = 0;
% save indic_gov indic_gov;
% [net,tr] = train(net,p,t);

indic_gov = 3;
save indic_gov indic_gov;
num_sim = 200;
p = [20*rand(1,num_sim)-10+100;10*rand(1,num_sim)-5-20;2*m0/100*rand(1,num_sim)-m0/100+m0;mfuel*ones(1,num_sim)];
t = ones(20,size(p,2));
[net,tr] = train(net,p,t);



