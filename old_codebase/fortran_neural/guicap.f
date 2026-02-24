c1
c1    copyright (c) AEROSPATIALE 1999
c1......................................................................
c2    nom    : guicap.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine la consigne de gite du guidage longitudinal en
c3    phase de capture (asservissement sur la trajectoire de reference)
c3
c3    NOTA  lorsque la gite equilibree ou la gite commandee ne sont pas
c3          definies(i.e. cos > 1), on force la gite commandee a prendre
c3          une valeur constante (a 0 ou a 90 deg ou encore a la gite
c3	    precedente actuellement).
c3......................................................................
c4    variables d'entree
c4
c4    positn(3)         R8    position absolue courante geocentrique
c4    vitesn(3)         R8    vitesse relative locale
c4    acceln(2)         R8    accelerations aerodynamiques estimees
c4    coefan(2)         R8    coefficients aerodynamiques estimes
c4    roguid            R8    densite atmospherique courante estimee
c4......................................................................
c5    variabels d'entree-sortie
c5
c5    iprepr(2)         I4    indicateur de securisation du guidage
c5......................................................................
c6    variables de sortie
c6
c6    gitlon            R8    consigne de gite guidage longitudinal
c6    iguida(2)         I4    indicateur de securisation du guidage
c6......................................................................
c7    variables internes
c7
c7    altitu            R8    altitude
c7    cosmuc            R8    cosinus de la gite commandee
c7    cosmue            R8    cosinus de la gite en vol equilibre
c7    gaindh            R8    gain d'oscillation en amplitude sur la vi
c7                            tesse radiale
c7    gainpd            R8    gain d'oscillation en amplitude sur la pres
c7                            sion dynamique
c7    imodel            I4    nuero de segment du profil de pdyn
c7    pdynrf            R8    pression dynamique sur profil de consigne
C7    rayvec            R8    rayon vecteur
c7    vitrad            R8    vitesse radiale
c7    vitrel            R8    vitesse relative
c7......................................................................
c8    composants appelants
c8
c8    guilon           INT    guidage longitudinal
c8......................................................................
c9    composants appeles
c9
c9    frayon           INT    determination du rayon de la planete
c9......................................................................
c10   commons utilises
c10
c10   capsul                  caracteristiques capsule
c10   carcap                  caracteristiques guidage longi equilibre
c10   geoide                  caracteristiques planete
c10   gravit                  caracteristiques gravitationnelles
c10   pdynln                  coefficeints profil de pdyn lineaire
c10   secgui
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  guicap (positn,vitesn,acceln,coefan,gitpre,roguid,
     +                    alfcom,
     +                    iprepr,vitref,
     +                    gitlon,iguida,temsim)
c
      implicit none
c
      integer  isecur,iprepr(2),iguida(2),
     +         iseccp,isecex,
     +         ntabul,ntabll, kinnrj,nbrnrj,itable,flag
     
      parameter (itable = 2000)
c
      double precision  positn(3),vitesn(3),acceln(2),coefan(2),roguid,
     +                  gitlon,gitpre,alfcom,
     +                  acgrav,altitu,amorth,cosmue,cosmuc,pdymax,
     +                  excent,degrad,gaindh,margmu,
     +                  pdyneq,pi,pulsah,rayvec,
     +                  srefer,vgitmx,vitrad,vitrel,xj2,xlatit,xmasse,
     +                  xmug,temsim,vitref,enrjlt,enrtot,coefpd(2),
     +                  nrjval, nrjpre, nrjhdt ,nrjtst,refhtt,
     +                  refpre, refhdt ,refincli ,refdates,refcmu,
     +                  hdtnom,httnom,prenom,cmunom,cosmu,gainhp,gainpd,
     +                  gitraf,penraf

      double precision tabepd(2,500,500),nbint2,interv
      
      double precision enrmin,nbint3,inter2
      
      double precision pdypre,pdycur,enrpre,enrcur,interp,intere
      
      double precision coeff1,coeff2,coeff3,coeff4,enrdeb
      
      double precision dalfae,disatm,dxdrag,dxlift
      
      integer i,j
      
      common / tabgit / tabepd
      common / csttab / nbint2,interv,enrmin,nbint3,inter2
      common / mecaer / dalfae,disatm,dxdrag,dxlift

      common / capsul / srefer,vgitmx,xmasse
      common / carcap / amorth,margmu(2),pulsah
      common / geoide / excent,xj2,xmug
      common / pargui / pdyneq
      common / secgui / iseccp,isecex
      common / trigon / degrad,pi 
      
      common / secpdy / pdymax    
c      
      common / nbgain / ntabul
      common / nbgite / ntabll  

      common / tabnrj / nrjval(itable), nrjpre(itable),nrjhdt(itable),
     +                  nrjtst(itable)
      common / reftab / refpre(itable), refhdt(itable),refhtt(itable),
     +                  refincli(itable),refdates(itable),refcmu(itable)
      common / colnrj / nbrnrj
      common / trajref / cosmu
      common / gains / gaindh,gainpd
      common / rafra2 / gitraf,penraf
      
c
      intrinsic  dabs,dacos,dsin,dtan,dcos
c
c		initialisations
c
      iguida(1) = 0
      iguida(2) = 0
c
      rayvec = positn(1)
      vitrel = vitesn(1)
      vitrad = vitesn(1)*dsin(vitesn(2))
      acgrav = xmug/positn(1)**2
c
c		energie totale
c
      enrjlt = enrtot (positn,vitesn)     
c
c		parametres de controle sur la trajectoire de reference
c         
      kinnrj = 2
      call  intrde (enrjlt,nrjval,refhtt,ntabul,kinnrj,
     +              httnom)
          
      kinnrj = 2
      call  intrde (enrjlt,nrjval,refpre,ntabul,kinnrj,
     +              prenom) 
          
      kinnrj = 2
      call  intrde (enrjlt,nrjval,refhdt,ntabul,kinnrj,
     +              hdtnom)
          
      kinnrj = 2
      call  intrde (enrjlt,nrjval,refcmu,ntabul,kinnrj,
     +              cmunom)
c
      call  frayon (positn,
     +              altitu,xlatit)

      pdyneq = 0.5d0*roguid*vitrel**2
c
c		gite en vol equilibre
c
      cosmue = cosmuc
c
c		commande en aerocapture (a securiser par max(pdyneq, 500)
c

c	altitu=vitrad*altitu/dabs(vitrad)
      
      flag=0
      
c  temporaire
      	interp=interv
      	intere=inter2
      	enrdeb=enrmin
      	
      	pdyneq=pdyneq/1000.
      	enrjlt=enrjlt/1000000.
      	pdypre=0.
      	pdycur=0.
      	enrpre=enrdeb+intere
      	enrcur=enrpre
      	i=0
      	j=0
      	
      	do while (((pdyneq-pdycur).ge.0).and.(i.lt.nbint2))
      		i=i+1
      		pdypre=pdycur
      		pdycur=pdycur+interp
      	end do
      	
      	do while (((enrjlt-enrcur).ge.0).and.(j.lt.nbint3))
      		j=j+1
      		enrpre=enrcur
      		enrcur=enrcur+intere
      	end do
      	
      	if (j.eq.0) then
      	gitraf = 0.
      	penraf = -11.89001713
      	else
      	coeff1=sqrt((pdycur-pdyneq)**2+(enrcur-enrjlt)**2)
      	coeff2=sqrt((pdycur-pdyneq)**2+(enrpre-enrjlt)**2)
      	coeff3=sqrt((pdypre-pdyneq)**2+(enrcur-enrjlt)**2)
      	coeff4=sqrt((pdypre-pdyneq)**2+(enrpre-enrjlt)**2)
	
	if (dabs(pdycur-pdyneq).lt.dabs(pdypre-pdyneq)) then
	  if (dabs(enrcur-enrjlt).lt.dabs(enrpre-enrjlt)) then
		gitraf=tabepd(1,i,j)
		penraf=tabepd(2,i,j)
c   	   	write(6,*) gitraf,i,j
	  else
	  	gitraf=tabepd(1,i,j-1)
		penraf=tabepd(2,i,j-1)
c     	 	write(6,*) gitraf,i,j-1
	  endif
	else
	  if (dabs(enrcur-enrjlt).lt.dabs(enrpre-enrjlt)) then
		gitraf=tabepd(1,i-1,j)
		penraf=tabepd(2,i-1,j)
c     	 	write(6,*) gitraf,i-1,j
	  else
	  	gitraf=tabepd(1,i-1,j-1)
		penraf=tabepd(2,i-1,j-1)
c   	   	write(6,*) gitraf,i-1,j-1
	  endif
	endif
	endif
	
c      	gitlon=coeff1*tabepd(1,i,j)+coeff2*tabepd(1,i,j-1)
c      	penraf=coeff1*tabepd(2,i,j)+coeff2*tabepd(2,i,j-1)
c      	gitlon=gitlon+coeff3*tabepd(1,i-1,j)+coeff4*tabepd(1,i-1,j-1)
c      	penraf=penraf+coeff3*tabepd(2,i-1,j)+coeff4*tabepd(2,i-1,j-1)
c      	gitlon=gitlon/(coeff1+coeff2+coeff3+coeff4)
c      	penraf=penraf/(coeff1+coeff2+coeff3+coeff4)
c      	gitlon=max(min((64.77026+(gitlon-64.77026)*3.2)/180.*pi,pi),0)
c      	gitraf=gitlon
c      	write(847,*) temsim,gitlon,enrjlt

      	pdyneq=pdyneq*1000.
      	enrjlt=enrjlt*1000000.
c  temporaire
      
      if (flag.eq.0) then	
      call  tbgain (altitu,coefan,alfcom,
     +              gaindh,gainpd,coefpd)
            
      cosmuc = cmunom + gaindh*(vitrad - hdtnom)/pdyneq
     +                + gainpd*(pdyneq - prenom)/pdyneq 
     
c
c			securisation du guidage
c
      if (dabs(cosmuc).gt.1.d0) then
         cosmuc = cosmuc/dabs(cosmuc)
         gitlon = dacos(cosmuc)
         isecur = 1
      else
         gitlon = dabs(dacos(cosmuc))
         isecur = 0
      endif 
c
c		incrementation du compteur de securisation du guidage
c
      if (isecur.eq.1) then
         iprepr(1) = iprepr(1) + 1
      endif
      
      else
      	interp=interv
      	intere=inter2
      	enrdeb=enrmin
      	
      	pdyneq=pdyneq/1000.
      	enrjlt=enrjlt/1000000.
      	pdypre=0.
      	pdycur=0.
      	enrpre=enrdeb
      	enrcur=enrdeb
      	i=0
      	j=0
      	
      	do while (((pdyneq-pdycur).ge.0).and.(i.lt.nbint2))
      		i=i+1
      		pdypre=pdycur
      		pdycur=pdycur+interp
      	end do
      	
      	do while (((enrjlt-enrcur).ge.0).and.(j.lt.nbint3))
      		j=j+1
      		enrpre=enrcur
      		enrcur=enrcur+intere
      	end do
      	
      	coeff1=1/sqrt((pdycur-pdyneq)**2+(enrcur-enrjlt)**2)
      	coeff2=1/sqrt((pdycur-pdyneq)**2+(enrpre-enrjlt)**2)
      	coeff3=1/sqrt((pdypre-pdyneq)**2+(enrcur-enrjlt)**2)
      	coeff4=1/sqrt((pdypre-pdyneq)**2+(enrpre-enrjlt)**2)

      	gitlon=coeff1*tabepd(1,i,j)+coeff3*tabepd(1,i,j-1)
      	gitlon=gitlon+coeff2*tabepd(1,i-1,j)+coeff4*tabepd(1,i-1,j-1)
      	gitlon=gitlon/(coeff1+coeff2+coeff3+coeff4)
      	gitlon=max(min((64.77026+(gitlon-64.77026)*3.2)/180.*pi,pi)
     +                  ,0.d0)
      	if (j.eq.0) then
      		gitlon=0.
      	endif
c      	write(6,*) gitlon

      	pdyneq=pdyneq*1000.
      	enrjlt=enrjlt*1000000.
      
      endif
      
      return
      end
