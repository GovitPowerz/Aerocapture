%function Trace_Corridor_MC_multi

h = 10;

figure

Nominal = 'nn_nom_gen2_new';
MonteCarlo = 'MC_1000_nn_gen2_171108';

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

grey = 0.9;

subplot(4,3,1)
hold on;
%plot(photo_nn(indices_fin(1),19)/1000000,photo_nn(indices_fin(1),20)/1000,'g+');
%plot(photo_nn(indices_fin(1),19)/1000000,photo_nn(indices_fin(1),20)/1000,'rx');
fill([visu_ovr(1,1);visu_ovr(:,1);visu_ovr(end,1)],[0;visu_ovr(:,2);0],[grey grey grey],'EdgeColor','k');
fill([5;5;visu_udr(1,1);visu_udr(:,1);visu_udr(end,1);-7;-7],[2.4;0;0;visu_udr(:,2);0;0;2.4],[grey grey grey],'EdgeColor','k');
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    s = [s 'photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',20)/1000,''b-'','];
end
s = [s 'photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',20)/1000,''b-'');'];
eval(s);
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,20)/1000,'r','LineWidth',1.5)
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    if (final(mm,43) < 150)
        s = [s 'photo_nn(' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm)) ',20)/1000,''g+'','];
    else
        s = [s 'photo_nn(' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm)) ',20)/1000,''rx'','];
    end
end
if (final(mm+1,43) < 150)
    s = [s 'photo_nn(' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm+1)) ',20)/1000,''g+'');'];
else
    s = [s 'photo_nn(' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm+1)) ',20)/1000,''rx'');'];
end
%eval(s);
xlabel('Orbital energy (MJ/kg)')
ylabel('Dynamic pressure (kPa)')
title('(a)')
%legend('Success','Failure','Location','NorthWest')
plot(-7,2.2,'k')
box on;
set(gca,'Layer','top');
grid on;
axis tight;
subplot(4,3,2)
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,10),'r','LineWidth',1.5)
hold on;
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    s = [s 'photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',10),''b-'','];
end
s = [s 'photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',10),''b-'');'];
eval(s);
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,10),'r','LineWidth',1.5)
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    if (final(mm,43) < 150)
        s = [s 'photo_nn(' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm)) ',10),''g+'','];
    else
        s = [s 'photo_nn(' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm)) ',10),''rx'','];
    end
end
if (final(mm+1,43) < 150)
    s = [s 'photo_nn(' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm+1)) ',10),''g+'');'];
else
    s = [s 'photo_nn(' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm+1)) ',10),''rx'');'];
end
%eval(s);
xlabel('Orbital energy (MJ/kg)')
ylabel('Inclination (deg)')
title('(b)')
%legend('Nominal','Monte Carlo','Location','NorthEast')
plot([-1;-1],[54.5 49.5],'w+')
box on;
set(gca,'Layer','top');
grid on;
axis tight;
subplot(4,3,3)
[y,x] = hist((final(:,43)),h);
bar(x,y/length(final(:,43)));
hi = findobj(gca,'Type','patch');
set(hi,'FaceColor','r','EdgeColor','k')
hold on;
[x,y] = cdfgov(final(:,43)');
semilogx(x,y);
title('(c)');
xlabel('Correction cost (m/s)');
ylabel('Distribution (-)');
axis([100 1500 0 1]);
set(gca,'Xscale','log','XTickLabel',{'100';'1000'})
grid on;

Nominal = 'nn_nom_gen19_new';
MonteCarlo = 'MC_1000_nn_gen19_171108';

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

subplot(4,3,4)
hold on;
fill([visu_ovr(1,1);visu_ovr(:,1);visu_ovr(end,1)],[0;visu_ovr(:,2);0],[grey grey grey],'EdgeColor','k');
fill([5;5;visu_udr(1,1);visu_udr(:,1);visu_udr(end,1);-7;-7],[2.4;0;0;visu_udr(:,2);0;0;2.4],[grey grey grey],'EdgeColor','k');
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    s = [s 'photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',20)/1000,''b-'','];
end
s = [s 'photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',20)/1000,''b-'');'];
eval(s);
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,20)/1000,'r','LineWidth',1.5)
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    if (final(mm,43) < 150)
        s = [s 'photo_nn(' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm)) ',20)/1000,''g+'','];
    else
        s = [s 'photo_nn(' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm)) ',20)/1000,''rx'','];
    end
end
if (final(mm+1,43) < 150)
    s = [s 'photo_nn(' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm+1)) ',20)/1000,''g+'');'];
else
    s = [s 'photo_nn(' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm+1)) ',20)/1000,''rx'');'];
end
%eval(s);
xlabel('Orbital energy (MJ/kg)')
ylabel('Dynamic pressure (kPa)')
title('(d)')
%legend('Undershoot','Overshoot','Nominal','Monte Carlo','Location','NorthWest')
plot(-7,2.2,'k')
box on;
set(gca,'Layer','top');
grid on;
axis tight;
subplot(4,3,5)
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,10),'r','LineWidth',1.5)
hold on;
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    s = [s 'photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',10),''b-'','];
end
s = [s 'photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',10),''b-'');'];
eval(s);
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,10),'r','LineWidth',1.5)
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    if (final(mm,43) < 150)
        s = [s 'photo_nn(' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm)) ',10),''g+'','];
    else
        s = [s 'photo_nn(' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm)) ',10),''rx'','];
    end
end
if (final(mm+1,43) < 150)
    s = [s 'photo_nn(' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm+1)) ',10),''g+'');'];
else
    s = [s 'photo_nn(' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm+1)) ',10),''rx'');'];
end
%eval(s);
xlabel('Orbital energy (MJ/kg)')
ylabel('Inclination (deg)')
title('(e)')
%legend('Nominal','Monte Carlo','Location','NorthEast')
plot([-1;-1],[54.5 49.5],'w+')
box on;
set(gca,'Layer','top');
grid on;
axis tight;
subplot(4,3,6)
[y,x] = hist((final(:,43)),h);
bar(x,y/length(final(:,43)));
hi = findobj(gca,'Type','patch');
set(hi,'FaceColor','r','EdgeColor','k')
hold on;
[x,y] = cdfgov(final(:,43)');
semilogx(x,y);
title('(c)');
xlabel('Correction cost (m/s)');
ylabel('Distribution (-)');
axis([100 1500 0 1]);
set(gca,'Xscale','log','XTickLabel',{'100';'1000'})
grid on;

Nominal = 'nn_nom_gen33_new';
MonteCarlo = 'MC_1000_nn_gen33_171108';

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

subplot(4,3,7)
hold on;
fill([visu_ovr(1,1);visu_ovr(:,1);visu_ovr(end,1)],[0;visu_ovr(:,2);0],[grey grey grey],'EdgeColor','k');
fill([5;5;visu_udr(1,1);visu_udr(:,1);visu_udr(end,1);-7;-7],[2.4;0;0;visu_udr(:,2);0;0;2.4],[grey grey grey],'EdgeColor','k');
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    s = [s 'photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',20)/1000,''b-'','];
end
s = [s 'photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',20)/1000,''b-'');'];
eval(s);
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,20)/1000,'r','LineWidth',1.5)
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    if (final(mm,43) < 150)
        s = [s 'photo_nn(' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm)) ',20)/1000,''g+'','];
    else
        s = [s 'photo_nn(' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm)) ',20)/1000,''rx'','];
    end
end
if (final(mm+1,43) < 150)
    s = [s 'photo_nn(' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm+1)) ',20)/1000,''g+'');'];
else
    s = [s 'photo_nn(' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm+1)) ',20)/1000,''rx'');'];
end
%eval(s);
xlabel('Orbital energy (MJ/kg)')
ylabel('Dynamic pressure (kPa)')
title('(g)')
%legend('Undershoot','Overshoot','Nominal','Monte Carlo','Location','NorthWest')
plot(-7,2.2,'k')
box on;
set(gca,'Layer','top');
grid on;
axis tight;
subplot(4,3,8)
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,10),'r','LineWidth',1.5)
hold on;
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    s = [s 'photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',10),''b-'','];
end
s = [s 'photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',10),''b-'');'];
eval(s);
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,10),'r','LineWidth',1.5)
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    if (final(mm,43) < 150)
        s = [s 'photo_nn(' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm)) ',10),''g+'','];
    else
        s = [s 'photo_nn(' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm)) ',10),''rx'','];
    end
end
if (final(mm+1,43) < 150)
    s = [s 'photo_nn(' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm+1)) ',10),''g+'');'];
else
    s = [s 'photo_nn(' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm+1)) ',10),''rx'');'];
end
%eval(s);
xlabel('Orbital energy (MJ/kg)')
ylabel('Inclination (deg)')
title('(h)')
%legend('Nominal','Monte Carlo','Location','NorthEast')
plot([-1;-1],[54.5 49.5],'w+')
box on;
set(gca,'Layer','top');
grid on;
axis tight;
subplot(4,3,9)
[y,x] = hist((final(:,43)),h);
bar(x,y/length(final(:,43)));
hi = findobj(gca,'Type','patch');
set(hi,'FaceColor','r','EdgeColor','k')
hold on;
[x,y] = cdfgov(final(:,43)');
semilogx(x,y);
title('(c)');
xlabel('Correction cost (m/s)');
ylabel('Distribution (-)');
axis([100 1500 0 1]);
set(gca,'Xscale','log','XTickLabel',{'100';'1000'})
grid on;

Nominal = 'nn_nom_new';
MonteCarlo = 'MC_1000_nn_171108';

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

subplot(4,3,10)
hold on;
fill([visu_ovr(1,1);visu_ovr(:,1);visu_ovr(end,1)],[0;visu_ovr(:,2);0],[grey grey grey],'EdgeColor','k');
fill([5;5;visu_udr(1,1);visu_udr(:,1);visu_udr(end,1);-7;-7],[2.4;0;0;visu_udr(:,2);0;0;2.4],[grey grey grey],'EdgeColor','k');
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    s = [s 'photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',20)/1000,''b-'','];
end
s = [s 'photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',20)/1000,''b-'');'];
eval(s);
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,20)/1000,'r','LineWidth',1.5)
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    if (final(mm,43) < 150)
        s = [s 'photo_nn(' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm)) ',20)/1000,''g+'','];
    else
        s = [s 'photo_nn(' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm)) ',20)/1000,''rx'','];
    end
end
if (final(mm+1,43) < 150)
    s = [s 'photo_nn(' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm+1)) ',20)/1000,''g+'');'];
else
    s = [s 'photo_nn(' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm+1)) ',20)/1000,''rx'');'];
end
%eval(s);
xlabel('Orbital energy (MJ/kg)')
ylabel('Dynamic pressure (kPa)')
title('(j)')
%legend('Undershoot','Overshoot','Nominal','Monte Carlo','Location','NorthWest')
plot(-7,2.2,'k')
box on;
set(gca,'Layer','top');
grid on;
axis tight;
subplot(4,3,11)
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,10),'r','LineWidth',1.5)
hold on;
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    s = [s 'photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',10),''b-'','];
end
s = [s 'photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',10),''b-'');'];
eval(s);
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,10),'r','LineWidth',1.5)
s = 'plot(';
for mm = 1:size(indices_deb,1)-1
    if (final(mm,43) < 150)
        s = [s 'photo_nn(' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm)) ',10),''g+'','];
    else
        s = [s 'photo_nn(' num2str(indices_fin(mm)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm)) ',10),''rx'','];
    end
end
if (final(mm+1,43) < 150)
    s = [s 'photo_nn(' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm+1)) ',10),''g+'');'];
else
    s = [s 'photo_nn(' num2str(indices_fin(mm+1)) ',19)/1000000,photo_nn(' num2str(indices_fin(mm+1)) ',10),''rx'');'];
end
%eval(s);
xlabel('Orbital energy (MJ/kg)')
ylabel('Inclination (deg)')
title('(k)')
%legend('Nominal','Monte Carlo','Location','NorthEast')
plot([-1;-1],[54.5 49.5],'w+')
box on;
set(gca,'Layer','top');
grid on;
axis tight;
subplot(4,3,12)
[y,x] = hist((final(:,43)),h);
bar(x,y/length(final(:,43)));
hi = findobj(gca,'Type','patch');
set(hi,'FaceColor','r','EdgeColor','k')
hold on;
[x,y] = cdfgov(final(:,43)');
semilogx(x,y);
title('(c)');
xlabel('Correction cost (m/s)');
ylabel('Distribution (-)');
axis([100 1500 0 1]);
set(gca,'Xscale','log','XTickLabel',{'100';'1000'})
grid on;

set(gcf,'PaperType','USLetter');
taille=get(gcf,'PaperSize');
set(gcf,'PaperSize',[taille(2) taille(1)]);
orient tall;
fn=['NN_MC_evol_2'];
print( gcf, '-dpng', fn );



