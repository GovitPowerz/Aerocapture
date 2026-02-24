g0 = 9.80665;
Isp = 228;
sref = 0.8;
cd = 1.5;
rho = 1.71e-8;
propmx = 1200;
tguid = 0.05;
tint = 0.25;
g = 3.718;
m0 = 160;
vf = [-5.0 0.0];

p = [100;0;-20;0];
n_gov = size(p,2);
t = 0;
y = [p(1,:)';p(2,:)';p(3,:)';p(4,:)';m0*ones(n_gov,1)];
dydt = y;
figure;
hold on;
while (max(abs(dydt)) > 0)
    plot(t,y(1),'r+',t,y(2),'b+')
%    plot(t,y(3),'r+',t,y(4),'b+')
%    plot(t,y(5),'b+')
    ground = (y(1:n_gov) > 0);
    burnout = (m0-y(4*n_gov+1:end) < 6);
    acc = [100 propmx./y(4*n_gov+1:end)];
    a_gov = max(min(acc,[],2),0);
    theta_gov = max(min(-pi/3,pi/4),-pi/4);
    vit_gov = sqrt(y(2*n_gov+1:3*n_gov).^2+y(3*n_gov+1:4*n_gov).^2);
    dydt = [y(2*n_gov+1:3*n_gov);y(3*n_gov+1:4*n_gov);...
        (a_gov.*cos(theta_gov).*burnout-g...
        -1/2*rho*sref*cd./y(4*n_gov+1:end).*vit_gov.*y(2*n_gov+1:3*n_gov));...
        (a_gov.*sin(theta_gov).*burnout...
        -1/2*rho*sref*cd./y(4*n_gov+1:end).*vit_gov.*y(3*n_gov+1:4*n_gov));...
        -y(4*n_gov+1:end).*a_gov/g0/Isp.*burnout].*[ground;ground;ground;ground;ground];
    y = y+tguid*dydt;
    t = t+tguid;
end
