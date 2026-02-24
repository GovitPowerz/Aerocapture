function Trace_Corridor_MC(Nominal,MonteCarlo)

load(['../sorties/photo.' MonteCarlo]);
photo_nn = photo;
load(['../sorties/final.' MonteCarlo]);
load(['../sorties/photo.' Nominal]);
photo_nn_nom = photo;
load visu.ovr_res
visu_ovr = visu;
load visu.udr_res
visu_udr = visu;

indices_deb = (find(diff(photo_nn(:,1)) < 0)+1);
indices_fin = [indices_deb-1;length(photo_nn(:,1))];
indices_deb = [1;indices_deb];

figure
plot(photo_nn_nom(:,1),photo_nn_nom(:,10),'r')
hold on;
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    s = [s 'photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',1),photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',10),''b-'','];
end
s = [s 'photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',1),photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',10),''b-'');'];
eval(s);
plot(photo_nn_nom(:,1),photo_nn_nom(:,10),'r')
xlabel('Time (s)')
ylabel('Inclination (deg)')
title('Inclination evolution')
legend('Nominal','Monte Carlo','Location','SouthEast')
grid on;
axis tight;

figure
plot(photo_nn_nom(:,1),photo_nn_nom(:,15),'r')
hold on;
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    s = [s 'photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',1),photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',15),''b-'','];
end
s = [s 'photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',1),photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',15),''b-'');'];
eval(s);
plot(photo_nn_nom(:,1),photo_nn_nom(:,15),'r')
xlabel('Time (s)')
ylabel('Bank angle (deg)')
title('Bank angle evolution')
legend('Nominal','Monte Carlo','Location','SouthEast')
grid on;
axis tight;


h = 20;
figure
[y,x] = hist(final(:,43),h);
bar(x,y/length(final(:,43)));
hi = findobj(gca,'Type','patch');
set(hi,'FaceColor','r','EdgeColor','k')
hold on;
[x,y] = cdfgov(final(:,43)');
plot(x,y);
title('Correction cost to reach parking orbit');
xlabel('(m/s)');
ylabel('Distribution (-)');
axis tight;
grid on;

grey = 0.9;

figure
subplot(2,2,1)
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,20)/1000,'r','LineWidth',1.5)
hold on;
%plot(visu_udr(:,1),visu_udr(:,2),'g','LineWidth',1.5)
%plot(visu_ovr(:,1),visu_ovr(:,2),'m','LineWidth',1.5)
%s = 'plot(';
%for mm = 1:size(indices_deb,1)-1
%    s = [s 'photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',20)/1000,''b-'','];
%end
%s = [s 'photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',20)/1000,''b-'');'];
%eval(s);
%plot(visu_udr(:,1),visu_udr(:,2),'g','LineWidth',1.5)
%plot(visu_ovr(:,1),visu_ovr(:,2),'m','LineWidth',1.5)
mm = 1;
plot(photo_nn(indices_deb(mm):indices_fin(mm),19)/1000000,photo_nn(indices_deb(mm):indices_fin(mm),20)/1000,'b-');
fill([visu_ovr(1,1);visu_ovr(:,1);visu_ovr(end,1)],[0;visu_ovr(:,2);0],[grey grey grey],'EdgeColor','k');
fill([5;5;visu_udr(1,1);visu_udr(:,1);visu_udr(end,1);-7;-7],[2.2;0;0;visu_udr(:,2);0;0;2.2],[grey grey grey],'EdgeColor','k');
for mm = 1:size(indices_deb,1)-1
    plot(photo_nn(indices_deb(mm):indices_fin(mm),19)/1000000,photo_nn(indices_deb(mm):indices_fin(mm),20)/1000,'b-');
end
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,20)/1000,'r','LineWidth',1.5)
xlabel('Orbital energy (MJ/kg)')
ylabel('Dynamic pressure (kPa)')
title('(a)')
%legend('Undershoot','Overshoot','Nominal','Monte Carlo','Location','NorthWest')
legend('Nominal','Monte Carlo','Location','NorthWest')
set(gca,'Layer','top');
box off;
grid on;
box on;
axis tight;
subplot(2,2,2)
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,10),'r','LineWidth',1.5)
hold on;
%s = 'plot(';
%for mm = 1:size(indices_deb,1)-1
%    s = [s 'photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',10),''b-'','];
%end
%s = [s 'photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',10),''b-'');'];
%eval(s);
for mm = 1:size(indices_deb,1)-1
    plot(photo_nn(indices_deb(mm):indices_fin(mm),19)/1000000,photo_nn(indices_deb(mm):indices_fin(mm),10),'b-');
end
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,10),'r','LineWidth',1.5)
xlabel('Orbital energy (MJ/kg)')
ylabel('Inclination (deg)')
title('(b)')
legend('Nominal','Monte Carlo','Location','NorthEast')
box off;
grid on;
box on;
axis tight;
subplot(2,2,3)
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,15),'r','LineWidth',1.5)
hold on;
%s = 'plot(';
%for mm = 1:size(indices_deb,1)-1
%    s = [s 'photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',15),''b-'','];
%end
%s = [s 'photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',15),''b-'');'];
%eval(s);
for mm = 1:size(indices_deb,1)-1
    plot(photo_nn(indices_deb(mm):indices_fin(mm),19)/1000000,photo_nn(indices_deb(mm):indices_fin(mm),15),'b-');
end
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,15),'r','LineWidth',1.5)
xlabel('Orbital energy (MJ/kg)')
ylabel('Bank angle (deg)')
title('(c)')
legend('Nominal','Monte Carlo','Location','SouthEast')
box off;
grid on;
box on;
axis tight;
subplot(2,2,4)
[y,x] = hist(final(:,43),h);
bar(x,y/length(final(:,43)));
hi = findobj(gca,'Type','patch');
set(hi,'FaceColor','r','EdgeColor','k')
hold on;
[x,y] = cdfgov(final(:,43)');
plot(x,y);
title('(d)');
xlabel('Correction cost (m/s)');
ylabel('Distribution (-)');
axis tight;
box off;
grid on;
box on;
set(gcf,'PaperType','USLetter');
taille=get(gcf,'PaperSize');
set(gcf,'PaperSize',[taille(2) taille(1)]);
orient tall;
fn=['NN_MC1000'];
print( gcf, '-dpng', fn );


function [xcdf,ycdf] = cdfgov(x)

n = length(x);
x = sort(x');
y = (1:n)'/n;
notdup = ([diff(x); 1] > 0);
x = x(notdup);
y = [0; y(notdup)];
k = length(x);
l = reshape(repmat(1:k, 2, 1), 2*k, 1);

xcdf = [-Inf; x(l); Inf];
ycdf = [0; 0; y(1+l)];

return;
