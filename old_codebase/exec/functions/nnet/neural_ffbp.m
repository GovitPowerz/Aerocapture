clear all;

addpath('nncontrol/');
addpath('nndemos/');
addpath('nnet/');
addpath('nnobsolete/');
addpath('nnutils/');

load input_train_rand.dat;
load output_train_rand.dat;

[pn,minp,maxp,tn,mint,maxt] = premnmx(input_train_rand(:,1:12)',output_train_rand');
%[pn,meanp,stdp,tn,meant,stdt] = prestd(input_train_rand',output_train_rand');

net = newff(minmax(pn),[20,14,1],{'tansig','tansig','tansig'},'trainlm');
net.trainParam.show = 1;
net.trainParam.epochs = 100;
net.trainParam.goal = 1e-6;
net.trainParam.mem_reduc = 2;

[net,tr] = train(net,pn,tn(1,:));
save nets_ffbp_sig_12_20_14 net;

net2 = newff(minmax(pn),[20,14,1],{'tansig','tansig','tansig'},'trainlm');
net2.trainParam.show = 1;
net2.trainParam.epochs = 100;
net2.trainParam.goal = 1e-6;
net2.trainParam.mem_reduc = 2;

[net2,tr2] = train(net2,pn,tn(2,:));
save nets_ffbp_sig_12_20_14 net net2;

net3 = newff(minmax(pn),[20,14,1],{'tansig','tansig','tansig'},'trainlm');
net3.trainParam.show = 1;
net3.trainParam.epochs = 100;
net3.trainParam.goal = 1e-6;
net3.trainParam.mem_reduc = 2;

[net3,tr3] = train(net3,pn,tn(3,:));
save nets_ffbp_sig_12_20_14 net net2 net3;
