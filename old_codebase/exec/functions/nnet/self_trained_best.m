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
g = 3.718;
m0 = 160;
vf = -5.0;
hf = 0.0;
mfuel = 3.2;
coef_opt = 1;
adim_gov = [3 18];
save prop_param m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov;

net = newff([0 200;-40 40;150 170;0 mfuel],[18,1],{'tansig','logsig'},'trainlm');

%init
net.trainParam.show = 1;
net.trainParam.epochs = 200;
net.trainParam.goal = 1e-12;
net.trainParam.mu_max = 1e12;
%net.performFcn = 'msereg';

% num_sim = 10000;
% p = [100 110 110 90 90 200*rand(1,num_sim);...
%     -20 -25 -15 -25 -15 80*rand(1,num_sim)-40;...
%     m0 m0 m0 m0 m0 2*m0/100*rand(1,num_sim)-m0/100+m0;...
%     mfuel mfuel mfuel mfuel mfuel mfuel*ones(1,num_sim)];
% t = 0.001*ones(1,size(p,2));
% indic_gov = 0;
% save indic_gov indic_gov;
% [net,tr] = train(net,p,t);

% indic_gov = 1;
% save indic_gov indic_gov;
% net.performFcn = 'mse';
% num_sim = 1;
% p = [100;-20;m0;mfuel];
% n_gov = 1;
% t = [];
% for i = 1:num_sim
%     disp(i);
%     a_gov = 0;
%     init = 1;
%     tgo = 30;
%     mem = zeros(3,4);
%     y = [p(1,i)';p(2,i)';p(3,i)'];
%     mass_ini = p(3,i)';
%     mass_fuel = p(4,i)';
%     dydt = y;
%     ndiv = 1;
%     count = 0;
%     while (max(abs(dydt)) > 0)
%         ground = (y(1:n_gov) > 10);
%         burnout = (mass_ini-y(2*n_gov+1:end) < mass_fuel);
%         [comopt,tgo,mem,init] = OPTIMAL_COMMAND([0.00001;0.00001;y(1)],[0.00001;0.00001;y(2)],a_gov,tgo,mem,init);
%         a_gov = min(max(comopt(3),0),propmx./y(2*n_gov+1:end));
%         dydt = [y(n_gov+1:2*n_gov); (a_gov.*burnout-g-1/2*rho*sref*cd_gov./y(2*n_gov+1:end).*y(n_gov+1:2*n_gov).*abs(y(n_gov+1:2*n_gov)));-y(2*n_gov+1:end).*a_gov/g0/Isp.*burnout].*[ground;ground;ground].*[burnout;burnout;burnout];
%         y = y+tguid*dydt;
%         if ((floor(count/ndiv) == count/ndiv)&&(count~=0))
%             p = [p [y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:end)';max(mass_fuel-(mass_ini-y(2*n_gov+1:end)),0)']];
%         end
%         if (floor(count/ndiv) == count/ndiv)
%             t = [t a_gov/propmx*y(2*n_gov+1:end)];
%         end
%         count = count + 1;
%     end
%     %     disp(y(n_gov+1:2*n_gov))
%     %     disp(mass_ini-y(2*n_gov+1:end))
% end
% [net,tr] = train(net,p,t);

indic_gov = 1;
save indic_gov indic_gov;
net.performFcn = 'mse';
num_sim = 10;
% figure;
% hold on;
train_record = [];
% net.trainParam.show = 1;
% net.trainParam.epochs = 10;
% net.trainParam.goal = 1e-9;
% net.trainParam.mu_max = 1e12;
% p = [20*rand(1,num_sim)-10+100;10*rand(1,num_sim)-5-20;2*m0/100*rand(1,num_sim)-m0/100+m0;mfuel*ones(1,num_sim)];
% t = ones(1,size(p,2));
% [net,tr] = train(net,p,t);
net.trainFcn = 'trainlm';
net.trainParam.show = 1;
net.trainParam.epochs = 100;
net.trainParam.goal = 1e-12;
net.trainParam.mu_max = 1e12;
net.trainParam.min_grad = 1e-14;
num_sim = 200;
vftmp = linspace(-15,5,4);
mftmp = linspace(1.6,3.6,10);
dvtmp = linspace(0.5,0.5,40);
dptmp = linspace(1.0,1.0,40);
dmtmp = linspace(1*m0/100,m0/100,40);
for i = 1:30
    %vf = vftmp(ceil(i/10));
%     mfuel = mftmp(ceil(i/2));
%     save prop_param m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov;
    p = [2*dptmp(i)*rand(1,num_sim)-dptmp(i)+100;...
         2*dvtmp(i)*rand(1,num_sim)-dvtmp(i)-20;...
         2*dmtmp(i)*rand(1,num_sim)-dmtmp(i)+m0;...
         mfuel*ones(1,num_sim)];
%     p = [100 110 110 90 90;...
%         -20 -25 -15 -25 -15;...
%         m0 m0 m0 m0 m0;...
%         mfuel mfuel mfuel mfuel mfuel];
    n_gov = size(p,2);
    t = 0;
    y = [p(1,:)';p(2,:)';p(3,:)'];
    mass_ini = p(3,:)';
    mass_fuel = p(4,:)';
    dydt = y;
    ndiv = 100000;
    count = floor((ndiv-1)*rand(1));
    while (max(abs(dydt)) > 0)
        ground = (y(1:n_gov) > 0);
        burnout = (mass_ini-y(2*n_gov+1:end) < mass_fuel);
        acc = sim(net,[y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:end)';max(mass_fuel-(mass_ini-y(2*n_gov+1:end)),0)'])';
        a_gov = acc.*propmx./y(2*n_gov+1:end);
        dydt = [y(n_gov+1:2*n_gov); (a_gov.*burnout-g-1/2*rho*sref*cd_gov./y(2*n_gov+1:end).*y(n_gov+1:2*n_gov).*abs(y(n_gov+1:2*n_gov)));-y(2*n_gov+1:end).*a_gov/g0/Isp.*burnout].*[ground;ground;ground].*[burnout;burnout;burnout];
        y = y+tguid*dydt;
        if ((floor(count/ndiv) == count/ndiv)&&(count~=0)&&(min(y(1:n_gov) > 10) > 0)&&(min(abs(dydt)) > 0))
            p = [p [y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:end)';max(mass_fuel-(mass_ini-y(2*n_gov+1:end)),0)']];
        end
        t = t+tguid;
        count = count + 1;
        %          plot(t,a_gov,'+');
    end
%     if (floor(i/3) == i/3)
%         vf = mean(y(n_gov+1:2*n_gov));
%         coef_opt = 1;
%         save prop_param m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov;
%     else
%         vf = -5;
%         coef_opt = 1;
%         save prop_param m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov;
%     end

%     t = ones(1,size(p,2));
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

    t = ones(1,size(p,2));
    [net,tr] = train(net,p,t);
    train_record = [train_record tr.perf];
    num_sim = num_sim + 0;
end

figure;
semilogy(train_record);


