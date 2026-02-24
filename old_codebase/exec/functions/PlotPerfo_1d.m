function PlotPerfo_1d(num_sim,t_mem,acc_mem,y_mem)

figure;
subplot(2,2,1);
plot(t_mem,acc_mem,'+');
grid on;
xlabel('time (s)');
ylabel('acceleration (m/s2)');
subplot(2,2,2);
plot(t_mem,y_mem(2*num_sim+1:end,:),'+');
grid on;
xlabel('time (s)');
ylabel('mass (kg)');
subplot(2,2,3);
plot(t_mem,y_mem(1:num_sim,:),'+');
grid on;
xlabel('time (s)');
ylabel('position (m)');
subplot(2,2,4);
plot(t_mem,y_mem(num_sim+1:2*num_sim,:),'+');
grid on;
xlabel('time (s)');
ylabel('velocity (m/s)');

