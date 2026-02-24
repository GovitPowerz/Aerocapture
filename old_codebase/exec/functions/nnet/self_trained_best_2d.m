clear all;

addpath('nncontrol/');
addpath('nndemos/');
addpath('nnet/');
addpath('nnobsolete/');
addpath('nnutils/');
warning off;
!rmd diary
diary on;

g0 = 9.80665;
Isp = 228;
sref = 0.8;
cd = 1.5;
rho = 1.71e-2;
propmx = 1200;
tguid = 0.05;
tint = 0.25;
g = 3.718;
m0 = 160;
vf = [-5.0 0.0];
mfuel = 10;
save prop_param m0 g0 Isp sref cd rho propmx tguid g vf;

net = network;
net.numInputs = 1;
net.numLayers = 6;
net.biasConnect = [1;1;1;1;1;1];
net.inputConnect = [1;1;0;0;0;0];
net.layerConnect(3,1) = 1;
net.layerConnect(4,2) = 1;
net.layerConnect(5,3) = 1;
net.layerConnect(6,4) = 1;
net.outputConnect = [0 0 0 0 1 1];
net.targetConnect = [0 0 0 0 1 1];
net.inputs{1}.range = [0 200;-2000 2000;-30 30;-100 100];
net.layers{1}.size = 28;
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.size = 28;
net.layers{1}.transferFcn = 'tansig';
net.layers{3}.size = 24;
net.layers{3}.transferFcn = 'tansig';
net.layers{4}.size = 24;
net.layers{4}.transferFcn = 'tansig';
net.layers{5}.transferFcn = 'purelin';
net.layers{6}.transferFcn = 'purelin';
net.performFcn = 'mse';
net.trainFcn = 'trainlm';

%net = newff([0 200;-2000 2000;-30 30;-100 100],[38,30,2],{'tansig','tansig','purelin'},'trainlm');
count_gov = 0;
while (count_gov < 40)
net.trainParam.show = 1;
net.trainParam.epochs = 20;
net.trainParam.goal = 1e-7;
net.trainParam.mu_max = 1e12;
p = [12*rand(1,100)-6+100;10*rand(1,100)-5;4*rand(1,100)-2-20;4*rand(1,100)-2];
n_gov = size(p,2);
t = 0;
y = [p(1,:)';p(2,:)';p(3,:)';p(4,:)';m0*ones(n_gov,1)];
dydt = y;
count = 0;
while ((max(abs(dydt)) > 0) && (count < 60))
    ground = (y(1:n_gov) > 0);
    burnout = (m0-y(4*n_gov+1:end) < mfuel);
    tmp = sim(net,[y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:3*n_gov)';y(3*n_gov+1:4*n_gov)'])';
    ax_gov = tmp(:,2);
    ay_gov = max(tmp(:,1),0);
    acc = sqrt((ax_gov).^2+(ay_gov).^2);
    a_gov = max(min(acc,propmx./y(4*n_gov+1:end)),0.00001);
    ax_gov = ax_gov./a_gov;
    ay_gov = ay_gov./a_gov;
    vit_gov = sqrt(y(2*n_gov+1:3*n_gov).^2+y(3*n_gov+1:4*n_gov).^2);
    dydt = [y(2*n_gov+1:3*n_gov);y(3*n_gov+1:4*n_gov);...
        (ay_gov.*burnout-g...
        -1/2*rho*sref*cd./y(4*n_gov+1:end).*vit_gov.*y(2*n_gov+1:3*n_gov));...
        (ax_gov.*burnout...
        -1/2*rho*sref*cd./y(4*n_gov+1:end).*vit_gov.*y(3*n_gov+1:4*n_gov));...
        -y(4*n_gov+1:end).*a_gov/g0/Isp.*burnout].*[ground;ground;ground;ground;ground];
    y = y+tguid*dydt;
    if ((floor(count/3) == count/3)&&(count~=0))
        p = [p [y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:3*n_gov)';y(3*n_gov+1:4*n_gov)']];
    end
    t = t+tguid;
    count = count + 1;
end
t = ones(2,size(p,2));
[net,tr] = train(net,p,t);
count_gov = count_gov+1;
end

net.trainParam.show = 1;
net.trainParam.epochs = 10;
net.trainParam.goal = 1e-7;
net.trainParam.mu_max = 1e12;
p = [12*rand(1,500)-6+100;10*rand(1,500)-5;4*rand(1,500)-2-20;4*rand(1,500)-2];
t = ones(2,size(p,2));
[net,tr] = train(net,p,t);

net.trainParam.show = 1;
net.trainParam.epochs = 10;
net.trainParam.goal = 1e-7;
net.trainParam.mu_max = 1e12;
p = [12*rand(1,1000)-6+100;10*rand(1,1000)-5;4*rand(1,1000)-2-20;4*rand(1,1000)-2];
t = ones(2,size(p,2));
[net,tr] = train(net,p,t);

net.trainParam.show = 1;
net.trainParam.epochs = 50;
net.trainParam.goal = 1e-7;
net.trainParam.mu_max = 1e12;
p = [12*rand(1,1500)-6+100;10*rand(1,1500)-5;4*rand(1,1500)-2-20;4*rand(1,1500)-2];
t = ones(2,size(p,2));
[net,tr] = train(net,p,t);

diary off;

