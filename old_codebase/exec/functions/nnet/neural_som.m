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
t = output_train_rand';

net = newsom([-1 1;-1 1;-1 1;0 2500;-1 1;-1 1;-1 1;0 100],[6 6 6]);
net.trainParam.show = 1;

net = train(net,p);
save net_som net;

