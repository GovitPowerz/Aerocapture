function dydt = f(t,y,n,net,m0,g0,Isp,sref,cd,rho,propmx,tguid,g)

ground = (y(1:n) > 0);
burnout = (m0-y(2*n+1:end) < 10);
acc = [sim(net,[y(1:n)';y(n+1:2*n)'])' propmx./y(2*n+1:end)];
a = max(min(acc,[],2),0);
dydt = [y(n+1:2*n); (a.*burnout-g-1/2*rho*sref*cd./y(2*n+1:end).*y(n+1:2*n).*abs(y(n+1:2*n)));-y(2*n+1:end).*a/g0/Isp.*burnout].*[ground;ground;ground];

