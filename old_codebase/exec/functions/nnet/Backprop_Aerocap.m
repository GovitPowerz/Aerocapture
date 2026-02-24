clear all;

addpath('nncontrol/');
addpath('nndemos/');
addpath('nnet/');
addpath('nnobsolete/');
addpath('nnutils/');
warning off;

% Initialisation des parametres physiques
load('../../fort.957');
input = fort;
load('../../fort.958');
output = fort;
save input_aerocap input;
save output_aerocap output;

net = newff([min(input);max(input)]',[16,2],{'asinhyp','sinhyp'},'trainlm');
tmp = getx(net);
net = setx(net,1e-5*rand(size(tmp)));
net = newff([min(input);max(input)]',[16,2],{'tansig','tansig'},'trainlm');

disp(' ');
disp('Entrainement du reseau...');
disp(' ');
% Training
train_record = [];
net.trainParam.show = 1;
net.trainParam.epochs = 20;
net.trainParam.goal = 1e-14;
net.trainParam.mu_max = 1e12;

for i = 1:100
	aleat = sortrows([rand(size(input,1),1) (1:size(input,1))']);
	indices = aleat(1:1e4,2);
	[net,tr] = train(net,input(indices,:)',output(indices,2:3)');
	train_record = [train_record tr.perf];
	semilogy(train_record);
	pause(0.05)
end

[net,tr] = train(net,input',output(:,2:3)');
nnet = getx(net);
save net_tansig_060707 net nnet;

figure;
semilogy(train_record);


