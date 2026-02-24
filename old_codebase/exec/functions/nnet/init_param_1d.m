
% Initialisation des paramètres physiques
g0 = 9.80665;
Isp = 228;
sref = 0.8;
cd_gov = 1.5;
rho = 1.71e-2;
propmx = 1200;
tguid = 0.1;
g = 3.718;
m0 = 152;
vf = -5.0;
hf = 0.0;
mfuel = 20;
coef_opt = 0.1;
adim_gov = [1 18];
alt_cut = 20;
mf_rest = mfuel;
save prop_param m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov mf_rest alt_cut;

% Conditions initiales nominales et dispersions correspondantes
posnom = 2000;
vitnom = -70;
masnom = m0;
dvtmp = 5;
dptmp = 10;
dmtmp = m0/100;
