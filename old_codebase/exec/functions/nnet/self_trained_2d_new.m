clear all;

addpath('nncontrol/');
addpath('nndemos/');
addpath('nnet/');
addpath('nnobsolete/');
addpath('nnutils/');
warning off;
diary on;

g0 = 9.80665;
Isp = 228;
sref = 0.8;
cd_gov = 1.5;
rho = 1.71e-2;
propmx = 1200;
tguid = 0.05;
tint = 0.25;
g = 3.718;
m0 = 160;
vf = -5.0;
hf = 0.0;
mfuel = 6;
coef_opt = 1;
adim_gov = [3 12 3 12];
save prop_param m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov;

net = newff([0 200;-100 100;-40 40;-40 40;150 170;0 mfuel],[18,2],{'tansig','logsig'},'trainbfg');

%init
net.trainParam.show = 1;
net.trainParam.epochs = 50;
net.trainParam.goal = 1e-12;
net.trainParam.mu_max = 1e12;

num_sim = 20000;
p = [200*rand(1,num_sim);200*rand(1,num_sim)-100;80*rand(1,num_sim)-40;80*rand(1,num_sim)-40;2*m0/100*rand(1,num_sim)-m0/100+m0;mfuel*ones(1,num_sim)];
t = [0.4*ones(1,size(p,2));0.5*ones(1,size(p,2))];
indic_gov = 0;
save indic_gov indic_gov;
[net,tr] = train(net,p,t);

indic_gov = 2;
save indic_gov indic_gov;
net.trainFcn = 'trainlm';
net.trainParam.show = 1;
net.trainParam.epochs = 50;
net.trainParam.goal = 1e-9;
net.trainParam.mu_max = 1e12;
train_record = [];
num_sim = 50;
for i = 1:60
    p = [20*rand(1,num_sim)-10+100;10*rand(1,num_sim)-5;10*rand(1,num_sim)-5-20;4*rand(1,num_sim)-2;2*m0/100*rand(1,num_sim)-m0/100+m0;mfuel*ones(1,num_sim)];
    n_gov = size(p,2);
    t = 0;
    y = [p(1,:)';p(2,:)';p(3,:)';p(4,:)';p(5,:)'];
    mass_ini = p(5,:)';
    mass_fuel = p(6,:)';
    dydt = y;
    ndiv = 20;
    count = floor((ndiv-1)*rand(1));
    while (max(abs(dydt)) > 0)
        ground = (y(1:n_gov) > 0);
        burnout = (mass_ini-y(4*n_gov+1:end) < mass_fuel);
        tmp = sim(net,[y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:3*n_gov)';...
            y(3*n_gov+1:4*n_gov)';y(4*n_gov+1:end)';...
            max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0)'])';
        ax_gov = 2*(tmp(:,2)-0.5);
        ay_gov = tmp(:,1);
        acc = sqrt((ax_gov).^2+(ay_gov).^2);
        a_gov = max(min(acc,1),1e-8);
        ax_gov = propmx*ax_gov.*a_gov./acc./y(4*n_gov+1:end);
        ay_gov = propmx*ay_gov.*a_gov./acc./y(4*n_gov+1:end);
        vit_gov = sqrt(y(2*n_gov+1:3*n_gov).^2+y(3*n_gov+1:4*n_gov).^2);
        dydt = [y(2*n_gov+1:3*n_gov);y(3*n_gov+1:4*n_gov);...
            (ay_gov.*burnout-g...
            -1/2*rho*sref*cd_gov./y(4*n_gov+1:end).*vit_gov.*y(2*n_gov+1:3*n_gov));...
            (ax_gov.*burnout...
            -1/2*rho*sref*cd_gov./y(4*n_gov+1:end).*vit_gov.*y(3*n_gov+1:4*n_gov));...
            -y(4*n_gov+1:end).*sqrt(ax_gov.^2+ay_gov.^2)/g0/Isp.*burnout].*[ground;ground;ground;ground;ground]...
            .*[burnout;burnout;burnout;burnout;burnout];
        y = y+tguid*dydt;
        if ((floor(count/ndiv) == count/ndiv)&&(count~=0)&&(min(y(1:n_gov) > 10) > 0)&&(min(abs(dydt)) > 0))
            p = [p [y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:3*n_gov)';y(3*n_gov+1:4*n_gov)';y(4*n_gov+1:end)';max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0)']];
        end
        t = t+tguid;
        count = count + 1;
    end
    if (floor(i/3) == i/3)
        vf = mean(y(2*n_gov+1:3*n_gov));
        coef_opt = 0;
        save prop_param m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov;
    else
        vf = -5;
        coef_opt = 1;
        save prop_param m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov;
    end

%     t = ones(2,size(p,2));
%     net_old = net;
%     net.trainParam.epochs = 1;
%     net.trainParam.goal = 1;
%     [net,tr] = train(net,p,t);
%     net = net_old;
%     net.trainParam.epochs = 50;
%     net.trainParam.goal = 0.9999*tr.perf(1);
%     [net,tr] = train(net,p,t);
%     
%     if (tr.perf(end) < net.trainParam.goal)
%         train_record = [train_record tr.perf];
%     else
%         net = net_old;
%     end

    t = ones(2,size(p,2));
    [net,tr] = train(net,p,t);
    train_record = [train_record tr.perf];
    num_sim = num_sim + 0;
end

semilogy(train_record);

diary off;

