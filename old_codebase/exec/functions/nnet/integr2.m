function err = integr2(net,p,m0,g0,Isp,sref,cd,rho,propmx,tguid,g,vf)

n = size(p,2);
tstart = 0;
tfinal = 20;
t = 0;
y = [p(1,:)';p(2,:)';m0*ones(n,1)];
dydt = y;
t_mem = [t];
y_mem = [y'];

while (max(abs(dydt)) > 0)
    dydt = f(t,y,n,net,m0,g0,Isp,sref,cd,rho,propmx,tguid,g);
    y = y+tguid*dydt;
    t = t+tguid;
%      t_mem = [t_mem;t];
%      y_mem = [y_mem;y'];
end
% plot(t_mem,y_mem(:,1:5));
% figure;
% plot(t_mem,y_mem(:,6:10));
% figure;
% plot(t_mem,y_mem(:,11:15));
signe = 2*((m0-y(2*n+1:end) < 10).*(y(n+1:2*n) < vf)-0.5);
err = 0.1*signe.*sqrt((y(n+1:2*n)-vf).^2+(1./exp(1.8*(10-m0+y(2*n+1:end)))).^2);
disp(' ');
disp('Propagation :');
disp(['erreurs position (moy,std) : ' num2str(mean(y(1:n))) ' ' num2str(std(y(1:n)))]);
disp(['erreurs vitesse (moy,std) : ' num2str(mean(y(n+1:2*n)-vf)) ' ' num2str(std(y(n+1:2*n)-vf))]);
disp(['consommation (moy,std) : ' num2str(mean(m0-y(2*n+1:end))) ' ' num2str(std(m0-y(2*n+1:end)))]);
