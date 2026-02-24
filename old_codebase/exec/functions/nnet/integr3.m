function err = integr2(net,p,m0,g0,Isp,sref,cd,rho,propmx,tguid,g)

options = odeset('Events',@events);
n = size(p,2)
tstart = 0;
tfinal = 100;
y0 = [p(1,:)';p(2,:)';m0*ones(n,1)];
[t,y,te,ye,ie] = ode45(@f,[tstart tfinal],y0,options,n,net,m0,g0,Isp,sref,cd,rho,propmx,tguid,g);
err = -200*sign(ye(n+1:2*n)).*sqrt(ye(1:n).^2+(ye(n+1:2*n)+1.0).^2+(m0-ye(2*n+1:end)).^2)/sqrt(100^2+20^2+10^2);

function [value,isterminal,direction] = events(t,y,n,net,m0,g0,Isp,sref,cd,rho,propmx,tguid,g)
% Locate the time when height passes through zero in a decreasing direction
% and stop integration.
value = sum(y(1:n) > 0)     % detect height = 0
isterminal = [1;1;1];   % stop the integration
direction = [0;0;0];   % negative direction
