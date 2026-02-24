clear all;

addpath('nncontrol/');
addpath('nndemos/');
addpath('nnet/');
addpath('nnobsolete/');
addpath('nnutils/');

load input_train_rand.dat;
load output_train_rand.dat;

%[pn,minp,maxp,tn,mint,maxt] = premnmx(input_train_rand',output_train_rand');
p = input_train_rand(:,1:8)';
t = output_train_rand(:,1)';

net = newlvq([-1 1;-1 1;-1 1;0 2500;-1 1;-1 1;-1 1;0 100],40);
net.trainParam.show = 1;

%tc = ind2vec(t);
net = train(net,p,t);
save net_lvq net;

