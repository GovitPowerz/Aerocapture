clear all;

addpath('nncontrol/');
addpath('nndemos/');
addpath('nnet/');
addpath('nnobsolete/');
addpath('nnutils/');

mass = 160;
sref = 0.8;
cd = 1.5;
rho = 1.71e-2;
propmx = 1200;
tguid = 0.25;
tint = 0.25;
g = 3.718;

net = newff([0 510;-31 0],[14,12,1],{'tansig','tansig','purelin'},'trainlm');
net.trainParam.show = 1;
net.trainParam.epochs = 1;
net.trainParam.goal = 1e-6;

p = [20*rand(1,200)-10+500;2*rand(1,200)-1-30];
for l = 1:2000
    a = sim(net,p);
    err = intrg(net,p,mass,sref,cd,rho,propmx,tguid,tint,g);
    t = a+err;
    [net,tr] = train(net,p,t);
end


