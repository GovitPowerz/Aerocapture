clear all;

addpath('nncontrol/');
addpath('nndemos/');
addpath('nnet/');
addpath('nnobsolete/');
addpath('nnutils/');
warning off;

% Initialisation des paramètres physiques
init_param_2d;

load net_1d_24_long;
net_1d = net;
net = newff([-500 500;0 2000;-50 50;-120 40;150 170],[48,2],{'tansig','logsig'},'trainlm');

num_sim = 500;
input = [2*dptmp(1)*rand(1,num_sim)-dptmp(1)+posnom(1);...
    2*dptmp(2)*rand(1,num_sim)-dptmp(2)+posnom(2);...
    2*dvtmp(1)*rand(1,num_sim)-dvtmp(1)+vitnom(1);...
    2*dvtmp(2)*rand(1,num_sim)-dvtmp(2)+vitnom(2);...
    2*dmtmp*rand(1,num_sim)-dmtmp+masnom;...
    mfuel*ones(1,num_sim)];

ndiv = 40;
nplot = 10;
[net] = init_net_2d(net,net_1d,input,ndiv,nplot);

num_sim = 1;
test_visu_net_2d(net,num_sim)

%save net_2d_init_32 net;
%load net_2d_init_48;

% Training
indic_gov = 2;
save indic_gov indic_gov;
train_record = [];
net.trainParam.show = 1;
net.trainParam.epochs = 25;
net.trainParam.goal = 1e-14;
net.trainParam.mu_max = 1e12;
coef = 0.1;
num_sim = 200;
for j = 1:4
    dvtmp = coef*[5;2.5];
    dptmp = coef*[30;100.0];
    dmtmp = m0/100;
    ndiv = 200;
    for i = 1:20
        p = [2*dptmp(1)*rand(1,num_sim)-dptmp(1)+2000;...
            2*dptmp(2)*rand(1,num_sim)-dptmp(2);...
            2*dvtmp(1)*rand(1,num_sim)-dvtmp(1)-70;...
            2*dvtmp(2)*rand(1,num_sim)-dvtmp(2);...
            2*dmtmp*rand(1,num_sim)-dmtmp+m0;...
            mfuel*ones(1,num_sim)];
        n_gov = size(p,2);
        t = 0;
        y = [p(1,:)';p(2,:)';p(3,:)';p(4,:)';p(5,:)'];
        mass_ini = p(5,:)';
        mass_fuel = p(6,:)';
        dydt = y;
        ndiv = ndiv+0;
        count = floor((ndiv-1)*rand(1));
        while (max(abs(dydt)) > 0)
            ground = (y(1:n_gov) > 0);
            burnout = (mass_ini-y(4*n_gov+1:end) < mass_fuel);
            zero_acc = 1-((y(2*n_gov+1:3*n_gov) > vf).*(y(1:n_gov) < alt_cut));
            zero_acc2 = 1-(y(1:n_gov) < alt_cut);
            stop_gov = (y(2*n_gov+1:3*n_gov) < vf/2);
            tmp = sim(net,[y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:3*n_gov)';...
                y(3*n_gov+1:4*n_gov)';y(4*n_gov+1:end)'])';
            ax_gov = 2*(0.5*tmp(:,2)-0.25);
            ay_gov = tmp(:,1);
            acc = max(sqrt((ax_gov).^2+(ay_gov).^2),1e-8);
            a_gov = max(min(acc,1),1e-8);
            ax_gov = propmx*ax_gov.*a_gov./acc./y(4*n_gov+1:end);
            ay_gov = g+(propmx*ay_gov.*a_gov./acc./y(4*n_gov+1:end)-g).*zero_acc;
            vit_gov = sqrt(y(2*n_gov+1:3*n_gov).^2+y(3*n_gov+1:4*n_gov).^2);
            dydt = [y(2*n_gov+1:3*n_gov);y(3*n_gov+1:4*n_gov);...
                (ay_gov.*burnout-g...
                -1/2*rho*sref*cd_gov./y(4*n_gov+1:end).*vit_gov.*y(2*n_gov+1:3*n_gov));...
                (ax_gov.*burnout...
                -1/2*rho*sref*cd_gov./y(4*n_gov+1:end).*vit_gov.*y(3*n_gov+1:4*n_gov));...
                -y(4*n_gov+1:end).*sqrt(ax_gov.^2+ay_gov.^2)/g0/Isp.*burnout].*[ground;ground;ground;ground;ground]...
                .*[burnout;burnout;burnout;burnout;burnout].*[stop_gov;stop_gov;stop_gov;stop_gov;stop_gov];;
            y = y+tguid*dydt;
            if ((floor(count/ndiv) == count/ndiv)&&(count~=0)&&(ndiv~=0)&&(min(y(1:n_gov) > 10) > 0)&&(min(abs(dydt)) > 0))
                p = [p [y(1:n_gov)';y(n_gov+1:2*n_gov)';y(2*n_gov+1:3*n_gov)';y(3*n_gov+1:4*n_gov)';y(4*n_gov+1:end)';max(mass_fuel-(mass_ini-y(4*n_gov+1:end)),0)']];
            end
            t = t+tguid;
            count = count + 1;
        end
        mf_rest = p(6,:);
        save prop_param m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov mf_rest alt_cut;

        t = ones(2,size(p,2));
        [net,tr] = train(net,p(1:5,:),t);
        train_record = [train_record tr.perf];
    end
     num_sim = num_sim+100;
   coef = coef*(10)^(1/3);
    net.trainParam.epochs = 2*net.trainParam.epochs;
end

figure;
semilogy(train_record);


