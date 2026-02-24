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
vf = -5.0;

net = newff([0 102;-21 0],[14,12,1],{'tansig','tansig','purelin'},'trainlm');

p = [102*rand(1,50000);-21*rand(1,50000)];
t = max(min(g+p(2,:).^2./(2*p(1,:)),propmx/(m0-10)),0);
net.trainParam.show = 1;
net.trainParam.epochs = 20;
net.trainParam.goal = 1e-6;
[net,tr] = train(net,p,t);

net.trainParam.epochs = 10;
net.trainParam.goal = 1e-6;
p = [];
%for k = 1:5
k=1;
    p = [p [4*rand(1,5000)-2+100;2*rand(1,5000)-1-20]];
    min = 10000;
    for l = 1:500
        disp(' ');
        disp(['case: ' num2str(k) ' ' num2str(l) ' ' num2str(min)]);
        tic;
        a = sim(net,p);
        toc;
        tic;
        err = integr2(net,p,m0,g0,Isp,sref,cd,rho,propmx,tguid,g,vf);
        t = a+err';
        net_old = net;
        toc;
        disp(' ');
        tic;
        [net,tr] = train(net,p,t);
        if (tr.perf(1) < min)
            min = tr.perf(1);
            save net_self net_old min;
        end
        save net_self_cur net
        toc;
    end
%end

diary off;

