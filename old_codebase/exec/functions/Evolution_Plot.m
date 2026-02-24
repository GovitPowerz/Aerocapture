function Evolution_Plot(Cout,cout_mem,ngen)

fig1=figure(1);
cla;
set(fig1,'Name',['Generation number: ' int2str(ngen)]);
set(fig1, 'Renderer', 'zbuffer');
% Range = [min(Cout);max(Cout)];
% Xvect=[];
% Yvect=[];
% Zvect=[];
% Nind=length(Cout(1,:));
% Nparam=length(Cout(:,1));
% 
% X=1:Nind;
% 
% for i=1:Nparam
%     Xvect=[Xvect X];
%     Yvect=[Yvect i.*ones(1,Nind)];
%     Zvect=[Zvect Cout(i,:)];
% end

% subplot(1,2,1);
% scatterbar3(Xvect,Yvect,Zvect,Range,0.5);
% axis square
% set(gca,'ZLim',[0*min(Range(1,:)) max(Range(2,:))]);
% xlabel('Individual number')
% ylabel('Subpopulation number')
% zlabel('Cost')
% title('dynamic view of parameters values')
% view(-130,30)
% colormap(jet(512))
% colorbar('peer',gca,'EastOutside','XLim',[0 0.1])
% subplot(1,2,2);
s = 'semilogy(';
for k = 1:size(cout_mem,1)-1
    s = [s '1:size(cout_mem,2),cout_mem(' num2str(k) ',:),'];
end
s = [s '1:size(cout_mem,2),cout_mem(' num2str(size(cout_mem,1)) ',:));'];
eval(s);
grid on;
axis tight;

drawnow;
