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
tint = 0.25;
g = 3.718;
m0 = 160;
vf = -5.0;
hf = 0.0;
mfuel = 3.1;
coef_opt = 1;
adim_gov = [20 0.8];
save prop_param m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov;

net = newff([0 200;-40 40;150 170;0 mfuel],[12,1],{'tansig','logsig'},'trainrp');

%init
net.trainParam.show = 25;
net.trainParam.epochs = 50;
net.trainParam.goal = 1e-0;
net.trainParam.min_grad = 1e-12;
net.trainParam.mu_max = 1e12;

% num_sim = 2000;
% p = [200*rand(1,num_sim);80*rand(1,num_sim)-40;2*m0/100*rand(1,num_sim)-m0/100+m0;mfuel*ones(1,num_sim)];
% t = 0.9*ones(1,size(p,2));
% indic_gov = 0;
% save indic_gov indic_gov;
% [net,tr] = train(net,p,t);

indic_gov = 1;
save indic_gov indic_gov;
num_sim = 50;
train_record = [];
for i = 1:50
    p = [20*rand(1,num_sim)-10+100;10*rand(1,num_sim)-5-20;2*m0/100*rand(1,num_sim)-m0/100+m0;mfuel*ones(1,num_sim)];
    n_gov = size(p,2);
    t = 0;
    y = [p(1,:)';p(2,:)';p(3,:)'];
    mass_ini = p(3,:)';
    mass_fuel = p(4,:)';
    dydt = y;
    ndiv = 1;
    count = (ndiv-1)*rand(1);
    while (min(abs(dydt)) > 0)
        ground = (y(1:n_gov) > 5);
        burnout = (mass_ini-y(2*n_gov+1:end) < mass_fuel);
        acc = sim(net,[y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:end)';max(mass_fuel-(mass_ini-y(2*n_gov+1:end)),0)'])';
        a_gov = acc.*propmx./y(2*n_gov+1:end);
        dydt = [y(n_gov+1:2*n_gov); (a_gov.*burnout-g-1/2*rho*sref*cd_gov./y(2*n_gov+1:end).*y(n_gov+1:2*n_gov).*abs(y(n_gov+1:2*n_gov)));-y(2*n_gov+1:end).*a_gov/g0/Isp.*burnout].*[ground;ground;ground].*[burnout;burnout;burnout];
        y = y+tguid*dydt;
        if ((floor(count/ndiv) == count/ndiv)&&(count~=0))
            p = [p [y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:end)';max(mass_fuel-(mass_ini-y(2*n_gov+1:end)),0)']];
        end
        t = t+tguid;
        count = count + 1;
    end

    t = ones(1,size(p,2));
    net_old = net;
    net.trainParam.epochs = 1;
    net.trainParam.goal = 1;
    [net,tr] = train(net,p,t);
    net = net_old;
    net.trainParam.epochs = 50;
    net.trainParam.goal = min(net.trainParam.goal,0.9999*tr.perf(1));
    [net,tr] = train(net,p,t);
    
    if (tr.perf(end) < net.trainParam.goal)
        train_record = [train_record tr.perf];
    else
        net = net_old;
    end
    num_sim = num_sim + 0;
end

semilogy(train_record);


