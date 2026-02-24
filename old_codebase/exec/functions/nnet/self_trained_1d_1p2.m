clear all;

addpath('nncontrol/');
addpath('nndemos/');
addpath('nnet/');
addpath('nnobsolete/');
addpath('nnutils/');
warning off;

% Initialisation des param??tres physiques
init_param_1d;

disp('Initialisation du reseau 1d...');
disp(' ');
net = newff([-pi pi;-pi pi;-pi pi],[5,1]);
net = newff([-pi/4 pi/4;-pi/4 pi/4;-pi/4 pi/4;-pi pi;-1 1;-1 1;-1 1],[32,3],{'tansig','tansig'},'trainlm');

num_sim = 500;
input = [2*dptmp*rand(1,num_sim)-dptmp+posnom;...
    2*dvtmp*rand(1,num_sim)-dvtmp+vitnom;...
    2*dmtmp*rand(1,num_sim)-dmtmp+masnom;...
    mfuel*ones(1,num_sim)];

ndiv = 10;
nplot = 10;
[net] = init_net_1d(net,input,ndiv,nplot);

%load net_1d_24_long;

disp('Visualisation des perfs du reseau 2d apr??s init...');
disp(' ');
num_sim = 10;
test_visu_net_1d(net,num_sim)

disp('Entrainement du reseau 2d...');
disp(' ');
% Training
indic_gov = 1;
save indic_gov indic_gov;
train_record = [];
net.trainFcn = 'trainrp';
net.trainParam.show = 1;
net.trainParam.epochs = 200;
net.trainParam.goal = 1e-14;
net.trainParam.mu_max = 1e12;
net.trainParam.min_grad = 1e-26;
coef = 1;
ndiv = 0;
num_sim = 50;
for j = 1:4
    dvtmp = coef*1.0;
    dptmp = coef*5.0;
    dmtmp = m0/100;
    for i = 1:20
        p = [2*dptmp*rand(1,num_sim)-dptmp+2000;...
            2*dvtmp*rand(1,num_sim)-dvtmp-70;...
            2*dmtmp*rand(1,num_sim)-dmtmp+m0;...
            mfuel*ones(1,num_sim)];
        n_gov = size(p,2);
        t = 0;
        y = [p(1,:)';p(2,:)';p(3,:)'];
        mass_ini = p(3,:)';
        mass_fuel = p(4,:)';
        dydt = y;
        ndiv = ndiv+20;
        count = floor((ndiv-1)*rand(1));
        while (max(abs(dydt)) > 0)
            ground = (y(1:n_gov) > 0);
            burnout = (mass_ini-y(2*n_gov+1:end) < mass_fuel);
            zero_acc = 1-((y(n_gov+1:2*n_gov) > vf).*(y(1:n_gov) < alt_cut));
            stop_gov = (y(n_gov+1:2*n_gov) < vf/2);
            acc = sim(net,[y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:end)'])';
            a_gov = g+(acc.*propmx./y(2*n_gov+1:end)-g).*zero_acc;
            dydt = [y(n_gov+1:2*n_gov); (a_gov.*burnout-g-1/2*rho*sref*cd_gov./y(2*n_gov+1:end).*y(n_gov+1:2*n_gov).*abs(y(n_gov+1:2*n_gov)));-y(2*n_gov+1:end).*a_gov/g0/Isp.*burnout].*[ground;ground;ground].*[burnout;burnout;burnout].*[stop_gov;stop_gov;stop_gov];
            y = y+tguid*dydt;
            if ((floor(count/ndiv) == count/ndiv)&&(count~=0)&&(ndiv~=0)&&(min(y(1:n_gov) > 10) > 0)&&(min(abs(dydt)) > 0))
                p = [p [y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:end)';max(mass_fuel-(mass_ini-y(2*n_gov+1:end)),0)']];
            end
            t = t+tguid;
            count = count + 1;
        end
        mf_rest = p(4,:);
        save prop_param m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov mf_rest alt_cut;

        t = ones(1,size(p,2));
        [net,tr] = train(net,p(1:3,:),t);
        train_record = [train_record tr.perf];
        num_sim = num_sim + 0;
    end
%    ndiv = ndiv+100000;
    num_sim = 2*num_sim;
    coef = coef*(10)^(1/3);
    net.trainParam.epochs = 2*net.trainParam.epochs;
end

figure;
semilogy(train_record);


