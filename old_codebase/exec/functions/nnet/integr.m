function err = integr(net,p,m0,g0,Isp,sref,cd,rho,propmx,tguid,g)

options = odeset('Events',@events);
for i = 1:size(p,2)
    tstart = 0;
    tfinal = 30;
    y0 = [p(1,i);p(2,i);m0];
    [t,y,te,ye,ie] = ode45(@f,[tstart tfinal],y0,options,net,m0,g0,Isp,sref,cd,rho,propmx,tguid,g);
    err(i) = -2*sign(y(end,2))*sqrt(y(end,1)^2+(y(end,2)+1.0)^2+(m0-y(end,3))^2)/sqrt(100^2+20^2+10^2);
end

function dydt = f(t,y,net,m0,g0,Isp,sref,cd,rho,propmx,tguid,g)

a = max(min(sim(net,[y(1);y(2)]),propmx/y(3)),0);
dydt = [y(2); (a-g-1/2*rho*sref*cd/y(3)*y(2)*abs(y(2)));-y(3)*a/g0/Isp];

% --------------------------------------------------------------------------

function [value,isterminal,direction] = events(t,y,net,m0,g0,Isp,sref,cd,rho,propmx,tguid,g)
% Locate the time when height passes through zero in a decreasing direction
% and stop integration.
value = [y(1);y(2)-20;m0-y(3)-10];     % detect height = 0
isterminal = [1;1;1];   % stop the integration
direction = [0;0;0];   % negative direction


