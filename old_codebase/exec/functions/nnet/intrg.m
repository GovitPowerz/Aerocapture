function err = intrg(net,p,mass,sref,cd,rho,propmx,tguid,tint,g)

for i = 1:size(p,2)
    postmp = p(1,i);
    vittmp = p(2,i);
    ttmp = 0.0;
%     plot(ttmp,postmp)
%     hold on;
    while ((postmp > 0) & (ttmp < 30) & (postmp < 2000) & (vittmp < 50))
        a = max(min(sim(net,[postmp;vittmp]),propmx/mass),0);
        for j = 1:tguid/tint
            postmp = postmp+vittmp*tint+1/2*(a-g-1/2*rho*sref*cd/mass*vittmp*abs(vittmp))^2*tint^2;
            vittmp = vittmp+tint*(a-g-1/2*rho*sref*cd/mass*vittmp*abs(vittmp));
            ttmp = ttmp+tint;
        end
%         plot(ttmp,postmp)
    end
%     pause
    err(i) = -10*sign(postmp)*sqrt(postmp^2+vittmp^2+(ttmp/1000)^2)/sqrt(2000^2+50^2+30^2);
end

