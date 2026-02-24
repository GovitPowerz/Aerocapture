function [ACC_LGT,TTGO_GUID,mem,init_out] = OPTIMAL_COMMAND(POS_VEH,VABS_VEH,THRUST_COM,Tgo_old,old,init)

%  Initialisations
clear m0 g0 Isp sref cd_gov rho propmx tguid g vf hf mfuel coef_opt adim_gov;
load prop_param;
altif      = 0.0;
alt_cut    = 0.0;
vT         = vf;
Tfin       = 0.0;
eps1       = 1.0;
eps2       = 0.1;
n_iter_max = 10;
posF = [0;0;0];
vitF = [0;0;0];
iter = 0;

% On vise un point au dessus du sol
gpla = [0;0;g];
posfin = [0;0;altif];
vitfin = [0;0;vT];
posest = POS_VEH;
vitest = VABS_VEH;
thrcomtot = THRUST_COM;

% Init
if (init == 1)
    init = 0;
    Tgo  = 2*sqrt(sum((posfin-posest).^2))/sqrt(sum((vitfin-vitest).^2));
    nu1  = 6*(2*(posest-posfin)+Tgo*(vitest+vitfin))/Tgo^3;
    nu2  = -2*(3*(posest-posfin)+Tgo*(vitest+2*vitfin))/Tgo^2-gpla;
    vitF = vitest-nu1*Tgo^2/2-(nu2+gpla)*Tgo;
    posF = posest+vitF*Tgo+nu1*Tgo^3/6+(nu2+gpla)*Tgo^2/2;

    Tgo_old = Tgo;
    old = [posF vitF nu1 nu2];
end

% CALCUL DES ACCELERATIONS COMMANDEES
if ((init == 0) && (Tgo_old > Tfin))
    if (thrcomtot > 0.001)
        lambda = sqrt(sum(old(:,4).^2))/thrcomtot;
        nu1 = old(:,3)/max(1,lambda);
        nu2 = old(:,4)/max(1,lambda);
    else
        nu1 = old(:,3);
        nu2 = old(:,4);
    end
    Tgo  = Tgo_old;
    posF = old(:,1);
    vitF = old(:,2);

    % ------ Initialisation de C : C = integrale(1/2 u**2 dt)/2
    cond = 0;
    iter = 0;

    % ------ Debut de la boucle sur la commande ------
    while ((cond == 0)&&(iter < n_iter_max))

        % ------ Calcul de psiF,OmegaF,uF ------
        psi1F   = posF - posfin;
        psi2F   = vitF - vitfin;
        uF      = -nu2;
        OmegaF  = sum(nu1.*vitF)-sum(nu2.*(gpla+nu2/2));

        % ------ Calcul des points initiaux atteints par integration a l'envers
        % ------ de la dynamique avec les parametres de guidage nui --------
        % ------ En effet u = - nu1 * (tf-t) - nu2                  --------
        pos1 = posF-Tgo*vitF-nu1*Tgo^3/6-(nu2+gpla)*Tgo^2/2;
        vit1 = vitF+nu1*Tgo^2/2+(nu2+gpla)*Tgo;

        % ------ Calcul des coefficients de la matrice reduite a inverser ------
        M = zeros(7,7);
        M = [...
            vitF(1)   vitF(2)   vitF(3)   uF(1)     uF(2)     uF(3)-gpla(3)   0;
            -Tgo^3/6   0         0        -Tgo^2/2   0         0              -(vitF(1)+nu1(1)*Tgo^2/2+nu2(1)*Tgo);
            0        -Tgo^3/6   0         0        -Tgo^2/2   0              -(vitF(2)+nu1(2)*Tgo^2/2+nu2(2)*Tgo);
            0         0        -Tgo^3/6   0         0        -Tgo^2/2        -(vitF(3)+nu1(3)*Tgo^2/2+(nu2(3)+gpla(3))*Tgo);
            Tgo^2/2   0         0         Tgo       0         0               nu1(1)*Tgo+nu2(1);
            0         Tgo^2/2   0         0         Tgo       0               nu1(2)*Tgo+nu2(2);
            0         0         Tgo^2/2   0         0         Tgo             nu1(3)*Tgo+nu2(3)+gpla(3)];

        % ------ Calcul de l'inverse de la matrice reduite  ------
        Minv = inv(M);

        % ------ Calcul des increments de commande pour atteindre le point vise ------
        % ---1-- Partie de la matrice inversible directement ------
        dposF = -eps1*psi1F;
        dvitF = -eps1*psi2F;

        % ------ Calcul des ecarts entre point atteint avec la commande courante et point vise ------
        d_pos = -eps1*(pos1-posest)-dposF+Tgo*dvitF;
        d_vit = -eps1*(vit1-vitest)-dvitF;
        d_ent = [-eps1*OmegaF-sum(nu1.*dvitF);d_pos;d_vit];

        % ---2-- Partie a inverser par Minv ------
        d_sor = Minv*d_ent;
        dnu1  = d_sor(1:3);
        dnu2  = d_sor(4:6);
        dTgo  = d_sor(7);

        norme = sqrt(sum(dposF.^2)+sum(dvitF.^2)+sum(dnu1.^2)+sum(dnu2.^2)+dTgo^2);
        if (norme <= eps2)
            cond = 1;
        else
            % ------ Calcul de la commande suivante ------
            posF = posF + dposF;
            vitF = vitF + dvitF;
            nu1  = nu1  + dnu1;
            nu2  = nu2  + dnu2;
            Tgo  = Tgo  + dTgo;
        end
        % ------ Fin de la boucle sur la convergence de la commande -------
        iter = iter+1;
    end
    % ------ Sauvegarde des variables ------
    mem = [posF vitF nu1 nu2];
else
    Tgo = 0.0;
    % ------ Sauvegarde des variables ------
    mem = old;
end

% sorties du programme
if ((iter < n_iter_max) && (Tgo > 0))
    TTGO_GUID = Tgo;
    ACC_LGT   = -nu2-nu1*Tgo;
else
    TTGO_GUID = Tgo_old;
    ACC_LGT   = -old(:,4)-old(:,3)*Tgo_old;
end
init_out  = init;

return;
