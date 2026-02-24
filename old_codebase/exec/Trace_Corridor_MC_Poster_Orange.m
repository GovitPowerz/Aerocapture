function Trace_Corridor_MC_Poster(Nominal,MonteCarlo)

load(['../sorties/photo.' MonteCarlo]);
photo_nn = photo;
load(['../sorties/final.' MonteCarlo]);
load(['../sorties/photo.' Nominal]);
photo_nn_nom = photo;
load visu.ovr_res
visu_ovr = visu;
load visu.udr_res
visu_udr = visu;

fontsize_reg = 16;

indices_deb = (find(diff(photo_nn(:,1)) < 0)+1);
indices_fin = [indices_deb-1;length(photo_nn(:,1))];
indices_deb = [1;indices_deb];

figure
set(gcf,'Color',[58 148 213]/256,'InvertHardCopy','off')
plot(photo_nn_nom(:,1),photo_nn_nom(:,10),'b','LineWidth',1.5)
hold on;
%s = 'plot(';
for mm = 1:size(indices_deb,1)
%for mm = 1
%    s = [s 'photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',1),photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',10),''Color'',[255 204 0]/256,'];
    plot(photo_nn(indices_deb(mm):indices_fin(mm),1),photo_nn(indices_deb(mm):indices_fin(mm),10),'-','Color',[208 100 1]/256)
end
%s = [s 'photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',1),photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',10),''-'',''Color'',[146 173 216]/256);']
%eval(s);
plot(photo_nn_nom(:,1),photo_nn_nom(:,10),'b','LineWidth',1.5)
xlabel('time from entry (s)')
ylabel('inclination (deg)')
title('inclination evolution')
%legend('Nominal','Monte Carlo','Location','SouthEast')
grid on;
axis tight;
set(gca,'Color',[33 29 119]/256, 'Xcolor','w', 'YColor','w','FontWeight','bold','Fontsize',fontsize_reg);
set(get(gca,'Title'),'Color','w','FontWeight','bold','Fontsize',fontsize_reg);
set(get(gca,'XLabel'),'Color','w','FontWeight','bold','Fontsize',fontsize_reg);
set(get(gca,'YLabel'),'Color','w','FontWeight','bold','Fontsize',fontsize_reg);
set(gcf,'PaperType','A5');
taille = get(gcf,'PaperSize');
set(gcf,'PaperSize',[taille(2) taille(1)]);
orient tall;
print(gcf,'-dpng','MC_Incli_Poster');

figure
subplot(2,1,1)
set(gcf,'Color',[58 148 213]/256,'InvertHardCopy','off')
plot(photo_nn_nom(:,1),cos(photo_nn_nom(:,15)*pi/180),'b','LineWidth',1.5)
hold on;
%s = 'plot(';
for mm = 1:size(indices_deb,1)
    %s = [s 'photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',1),photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',15),''b-'','];
    plot(photo_nn(indices_deb(mm):indices_fin(mm),1),cos(photo_nn(indices_deb(mm):indices_fin(mm),15)*pi/180),'-','Color',[208 100 1]/256)
end
%s = [s 'photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',1),photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',15),''-'',''Color'',[146 173 216]/256);'];
%eval(s);
plot(photo_nn_nom(:,1),cos(photo_nn_nom(:,15)*pi/180),'b','LineWidth',1.5)
ylabel('cosine (-)')
title('bank angle evolution')
%legend('Nominal','Monte Carlo','Location','SouthEast')
grid on;
axis([photo_nn_nom(1,1) photo_nn_nom(end,1) -1 1]);
set(gca,'Color',[33 29 119]/256, 'Xcolor','w', 'YColor','w','FontWeight','bold','Fontsize',fontsize_reg);
set(get(gca,'Title'),'Color','w','FontWeight','bold','Fontsize',fontsize_reg);
set(get(gca,'XLabel'),'Color','w','FontWeight','bold','Fontsize',fontsize_reg);
set(get(gca,'YLabel'),'Color','w','FontWeight','bold','Fontsize',fontsize_reg);
subplot(2,1,2)
set(gcf,'Color',[58 148 213]/256,'InvertHardCopy','off')
plot(photo_nn_nom(:,1),sin(photo_nn_nom(:,15)*pi/180),'b','LineWidth',1.5)
hold on;
%s = 'plot(';
for mm = 1:size(indices_deb,1)
    %s = [s 'photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',1),photo_nn(' num2str(indices_deb(mm)) ':' num2str(indices_fin(mm)) ',15),''b-'','];
    plot(photo_nn(indices_deb(mm):indices_fin(mm),1),sin(photo_nn(indices_deb(mm):indices_fin(mm),15)*pi/180),'-','Color',[208 100 1]/256)
end
%s = [s 'photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',1),photo_nn(' num2str(indices_deb(mm+1)) ':' num2str(indices_fin(mm+1)) ',15),''-'',''Color'',[146 173 216]/256);'];
%eval(s);
plot(photo_nn_nom(:,1),sin(photo_nn_nom(:,15)*pi/180),'b','LineWidth',1.5)
xlabel('time from entry (s)')
ylabel('sine (-)')
%legend('Nominal','Monte Carlo','Location','SouthEast')
grid on;
axis([photo_nn_nom(1,1) photo_nn_nom(end,1) -1 1]);
set(gca,'Color',[33 29 119]/256, 'Xcolor','w', 'YColor','w','FontWeight','bold','Fontsize',fontsize_reg);
set(get(gca,'Title'),'Color','w','FontWeight','bold','Fontsize',fontsize_reg);
set(get(gca,'XLabel'),'Color','w','FontWeight','bold','Fontsize',fontsize_reg);
set(get(gca,'YLabel'),'Color','w','FontWeight','bold','Fontsize',fontsize_reg);
set(gcf,'PaperType','A5');
taille = get(gcf,'PaperSize');
set(gcf,'PaperSize',[taille(2) taille(1)]);
orient tall;
print(gcf,'-dpng','MC_Bank_Poster');

figure
set(gcf,'Color',[58 148 213]/256,'InvertHardCopy','off')
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,20)/1000,'b','LineWidth',1.5)
hold on;
mm = 1;
plot(photo_nn(indices_deb(mm):indices_fin(mm),19)/1000000,photo_nn(indices_deb(mm):indices_fin(mm),20)/1000,'b-');
fill([visu_ovr(1,1);visu_ovr(:,1);visu_ovr(end,1)],[0;visu_ovr(:,2);0],[146 173 216]/256,'EdgeColor','k');
fill([5;5;visu_udr(1,1);visu_udr(:,1);visu_udr(end,1);-7;-7],[2.2;0;0;visu_udr(:,2);0;0;2.2],[146 173 216]/256,'EdgeColor','k');
for mm = 1:size(indices_deb,1)-1
    plot(photo_nn(indices_deb(mm):indices_fin(mm),19)/1000000,photo_nn(indices_deb(mm):indices_fin(mm),20)/1000,'-','Color',[208 100 1]/256);
end
plot(photo_nn_nom(:,19)/1000000,photo_nn_nom(:,20)/1000,'b','LineWidth',1.5)
xlabel('orbital energy (MJ/kg)')
ylabel('dynamic pressure (kPa)')
%legend('Undershoot','Overshoot','Nominal','Monte Carlo','Location','NorthWest')
%legend('Nominal','Monte Carlo','Location','NorthWest')
set(gca,'Layer','top');
box off;
grid on;
box on;
axis tight;
set(gca,'Color',[33 29 119]/256, 'Xcolor','w', 'YColor','w','FontWeight','bold','Fontsize',fontsize_reg);
set(get(gca,'Title'),'Color','w','FontWeight','bold','Fontsize',fontsize_reg);
set(get(gca,'XLabel'),'Color','w','FontWeight','bold','Fontsize',fontsize_reg);
set(get(gca,'YLabel'),'Color','w','FontWeight','bold','Fontsize',fontsize_reg);
set(gcf,'PaperType','A5');
taille = get(gcf,'PaperSize');
set(gcf,'PaperSize',[taille(2) taille(1)]);
orient tall;
print(gcf,'-dpng','MC_Corridor_Poster');



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
