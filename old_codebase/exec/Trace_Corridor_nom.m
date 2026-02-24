load ../sorties/photo.nn_nom_gen142_new
photo_nn = photo;
load ../sorties/photo.ftc_nom_new
photo_ftc = photo;
load visu.ovr_res
visu_ovr = visu;
load visu.udr_res
visu_udr = visu;

grey = 0.9;

figure
subplot(2,1,1)
%plot(visu_udr(:,1),visu_udr(:,2),'g','LineWidth',1.5)
hold on;
%plot(visu_ovr(:,1),visu_ovr(:,2),'m','LineWidth',1.5)
plot(photo_ftc(:,19)/1000000,photo_ftc(:,20)/1000,'r','LineWidth',1.5)
plot(photo_nn(:,19)/1000000,photo_nn(:,20)/1000,'b','LineWidth',1.5)
fill([visu_ovr(1,1);visu_ovr(:,1);visu_ovr(end,1)],[0;visu_ovr(:,2);0],[grey grey grey],'EdgeColor','k');
fill([5;5;visu_udr(1,1);visu_udr(:,1);visu_udr(end,1);-7;-7],[2.2;0;0;visu_udr(:,2);0;0;2.2],[grey grey grey],'EdgeColor','k');
xlabel('Orbital energy (MJ/kg)')
ylabel('Dynamic pressure (kPa)')
title('(a)')
%legend('Undershoot','Overshoot','FTC','Neural','Location','NorthWest')
legend('FTC','Neural','Location','NorthWest')
plot(-7,2.2,'k')
box on;
set(gca,'Layer','top');
grid on;
axis([-7 5 0 2.2]);
subplot(2,2,3)
plot(photo_ftc(:,19)/1000000,photo_ftc(:,10),'r','LineWidth',1.5)
hold on;
plot(photo_nn(:,19)/1000000,photo_nn(:,10),'b','LineWidth',1.5)
xlabel('Orbital energy (MJ/kg)')
ylabel('Inclination (deg)')
title('(b)')
legend('FTC','Neural','Location','NorthEast')
grid on;
axis tight;
subplot(2,2,4)
plot(photo_ftc(:,19)/1000000,photo_ftc(:,15),'r','LineWidth',1.5)
hold on;
plot(photo_nn(:,19)/1000000,photo_nn(:,15),'b','LineWidth',1.5)
xlabel('Orbital energy (MJ/kg)')
ylabel('Bank angle (deg)')
title('(c)')
legend('FTC','Neural','Location','NorthEast')
grid on;
axis tight;
set(gcf,'PaperType','USLetter');
taille=get(gcf,'PaperSize');
set(gcf,'PaperSize',[taille(2) taille(1)]);
orient tall;
fn=['Nom_comp2'];
print( gcf, '-dpng', fn );
