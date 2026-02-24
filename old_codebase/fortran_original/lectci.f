c1
c1    copyright (c) AEROSPATIALE 1999
c1......................................................................
c2    nom    : lectci.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module realise la lecture des fichiers de donnees de simulation
c3    ainsi que la fermeture de ceux-ci apres utilisation. Les donnees
c3    lues dans ces fichiers sont sauvegardees dans des commons.
c3    On initialise egalement certaines constantes sauvegardees dans des
c3    commons.
c3
c3    NOTA  On suppose les erreurs de navigation en debut d'aerocapture 
c3          et courantes geocentriques.
c3.......................................................................
c6    variables de sortie
c6
c6    iconfd            I4   indicateur de confirmation des choix
c6......................................................................
c8    composants appelants
c8
c8    cisimu            INT  conditions generales de simulation
c8......................................................................
c9    composants appeles
c9
c9    frayon            INT  rayon planete
c9    geodes            INT  position geocentrique
c9......................................................................
c10   commons utilises
c10
c10   aernom
c10   finnom            R8    finesse nominale
c10
c10   capsul
c10   srefer            R8   surface de reference
c10   vgitmx            R8   vitesse de gite maximale
c10   xmasse            R8   masse capsule
c10
c10   congui
c10   pdacti            R8   pression dynamique d'activation guidage
c10   pdinib            R8   pression dynamqiue d'inhibition guidage
c10
c10   conren
c10   conacc            R8   contrainte de facteur de charge
c10   conflu            R8   contrainte de flux
c10   conpdy            R8   contrainte de pression dynamique
c10
c10   disaer
c10   droatm            R8   meconnaissance densite atmospherique
c10   dxdrag            R8   meconnaissance coef. trainee
c10   dxlift            R8   meconnaissance coef. portance
c10
c10   disini
c10   daltzd            R8   dispersion initiale altitude
c10   dlonzd            R8   dispersion initiale longitude
c10   dlatzd            R8   dispersion initiale latitude
c10   dvitzd            R8   dispersion initiale norme vitesse
c10   dazmzd            R8   dispersion initiale azimut vitesse
c10   dpenzd            R8   dispersion initiale pente vitesse
c10
c10   disnav
c10   dnaval            R8   erreur navigation altitude
c10   dnavlo            R8   erreur navigation longitude
c10   dnavla            R8   erreur navigation latitude
c10   dnavvi            R8   erreur navigation norme vitesse
c10   dnavpe            R8   erreur navigation pente vitesse
c10   dnavaz            R8   erreur navigation azimut vitesse
c10   dnavad            R8   erreur mesure acceleration de portance
c10
c10   estiro
c10   lambda            R8  coefficient pour estimation de densite
c10
c10   geoide
c10   excent            R8   excentrcite planete
c10   xj2               R8   coefficient premier harmonique J2 planete
c10   xmug              R8   constante gravitationnelle planete
c10
c10   gravit
c10   g0terr            R8   acceleration de pesanteur terrestre
c10   g0mars            R8   acceleration de pesanteur planete
c10
c10   loglat
c10   enrlat            R8   seuil d'activation du guidage lateral
c10
c10   missio
c10   xaltfn            R8   altitude finale visee
c10   xlonfn            R8   longitude finale visee
c10   xlatfn            R8   latitude finale visee
c10   xvitfn            R8   vitesse finale visee
c10   xpenfn            R8   pente vitesse finale visee
c10   xazmfn            R8   azimut vitesse finale visee
c10
c10   modatm
c10   cstgam            R8   constante des gaz
c10   facech            R8   facteur d'echelle modele exponentiel
c10   rozmod            R8   pression de reference modele
c10   rmoyen            R8   rayon moyen planete
c10
c10   modcon
c10   inibac            I4   indicateur d'inhibition contrainte facteur de
c10                          charge
c10   inibfl            I4   indicateur d'inhibition contrainte de flux
c10                          thermique
c10
c10   modven
c10   ivents            I4   indicateur de presence de vents
c10
c10   nrjvis
c10   enrjfn            R8   energie finale visee
c10   vitzfn            R8   vitesse verticale finale visee
c10
c10   orbvis
c10   zapoge            R8   apoastre vise
c10   zperig            R8   periastre vise
c10   demiax            R8   demi grand axe vise
c10   excorb            R8   excentricite visee
c10   xincli            R8   inclinaison visee
c10   gomega            R8   longitude noeud ascendant visee
c10
c10   parkin
c10   zapotf            R8   alittude apoastre orbite de parking
c10   zpertf            R8   altitude periastre orbite de parking
c10
c10   period
c10   tnavig            R8   cadence navigation
c10   tguida            R8   cadence guidage
c10   tpilot            R8   cadence pilote
c10   tpredi            R8   cadence prediction
c10   tinteg            R8   cadence elementaire d'integration
c10
c10   planet
c10   xomega(3)         R8   vitesse de rotation planete
c10   rmoyen            R8   rayon moyen planete
c10   requat            R8   rayon equatorial planete
c10   rpolar            R8   rayon polaire planete
c10
c10   profro
c10   xaltro(5)         R8   altitudes du gabarit de dispersion
c10   xdisro(5,2)       R8   valeurs max de dispersion en %
c10
c10   satorb
c10   enrmin            R8   seuil d'energie pour passage en parabole
c10
c10   tabaer
c10   tabcxe(nmachx)    R8   coefficients de trainee  equilibres
c10   tabcze(nmachx)    R8   coefficients de portance equilibres
c10
c10   tablar
c10   nbmach            I4   nombre de points de Mach des tables aerody
c10                          namiques
c10
c10   tablat
c10   nbalti            I4   nombre de points d'altitude des tables at-
c10                          mospheriques
c10
c10   tkodak
c10   tphoto            R8   cadence d'instantannees sur la trajectoire
c10
c10   trigon
c10   degrad            R8   conversion degres-radians
c10   pi                R8   nombre pi
c10
c10   vlimit
c10   epsiln            R8   seuil de comparaison
c10
c10   xvrent
c10   positz(3)         R8   position nominale debut aerocapture
c10   vitesz(3)         R8   vitesse nominale debut aerocapture
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  lectci  (iconfd)
c
      implicit none
c
      include '../include/dimensions.incl'
c
      integer  iconfd,
     +         i,inibac,inibfl,ivents,kintcx,kintcz,nbalfa,nbalti,
     +         nbmach,irevrs,iseccp,isecex,ntabul,naltit,natman,
     +	       ntabll,nbrnrj,natpla,irefer,itable,nzapd,nzapdmx,
     +         natpil,atmver,atmvar
     
      parameter (itable = 2000)
      
      parameter (nzapdmx = 1000)
c
      double precision  alfaeq,alfpre,altcst,amorth,conacc,
     +                  conflu,conpdy,coridx,coridy,cstgam,cxenom,
     +                  czenom,daltzd,datini,dazmzd,degrad,demiax,
     +                  dincid,dlonzd,dlatzd,dnavad,dnaval,dnavlo,
     +                  dnavla,dnavvi,dnavpe,dnavaz,dpenzd,droatm,
     +                  dvitzd,dxdrag,dxlift,dzalim,enrjfn,enrlat,
     +                  enrmin,epsiln,errinc,errvit,errzap,errzpe,
     +                  excent,excorb,facech,finnom,gaindh,
     +                  gitpre,gomega,g0terr,g0mars,lambda,margmu,
     +                  pdacti, pdinib,pi,positz,profax,profay,
     +                  pulsah,requat,rmoyen,rozmod,rpolar,
     +                  srefer,tabcae(nmachx),tabcne(nmachx),
     +                  tabcxe,tabcze,tabmac,tguida,
     +                  tinteg,tnavig,tphoto,tpilot,tpredi,vgitmx,
     +                  vitesz,vitzfn,vsorti,xaltfn,xaltzd,xaltro,
     +                  xazmfn,xazmzd,xdisro(5),xgabro,xlatfn,xlatzd,
     +                  xlonfn,xlonzd,xpenfn,xpenzd,xvitfn,xvitzd,
     +                  xincli,xj2,xmasse,xmug,amorft,pulsft,
     +                  xomega,zapoge,zapotf,zperig,zpertf,
     +                  zromod,altatm,romatm,cinexi,dlong,
     +                  ks,dlat,dvit,dpen,dazm,dcx,dcz,pdymax,
     +			dro,nrjval,refpre,refhdt,nrjtst,refhtt,
     +                  refincli,refdates,nrjpre,nrjhdt,ballistique,
     +                  refcmu,cq,gitref,altpdn,tabpda,tabpdb,xmulti,
     +                  xvitinf,xaltzp,xazmabs,xr,xrp,ex,nu,p,v,vx,
     +                  vy,vz,vpxy,anoinf,vitinf,dmasse,cstpil,amrpil,
     +                  omgpil,atmdis,ampli,wavlen,pente

c
      common / aeroeq / alfaeq
      common / aernom / cxenom,czenom,finnom
      common / capsul / srefer,vgitmx,xmasse
      common / carcap / amorth,margmu(2),pulsah
      common / carext / altcst,dzalim,gaindh
      common / congui / pdacti,pdinib
      common / conren / conacc,conflu,conpdy
      common / corrid / coridx,coridy
      common / disaer / dincid,droatm,dxdrag,dxlift
      common / disini / daltzd,dlonzd,dlatzd,dvitzd,dazmzd,dpenzd
      common / dismas / dmasse
      common / disnav / dnaval,dnavlo,dnavla,dnavvi,dnavpe,dnavaz,
     +                  dnavad
      common / estiro / lambda
      common / geoide / excent,xj2,xmug
      common / gravit / g0terr,g0mars
      common / loglat / enrlat(2)
      common / loialf / profax(nalfax),profay(nalfax)
      common / missio / xaltfn,xlonfn,xlatfn,xvitfn,xpenfn,xazmfn
      common / misaga / anoinf,vitinf
      common / modaga / natman
      common / modalf / nbalfa
      common / modatm / cstgam,facech,rozmod,rmoyen,zromod
      common / modcon / inibac,inibfl
      common / modven / ivents
      common / muldis / xmulti(4)
      common / nrjvis / enrjfn,vitzfn
      common / orbvis / zapoge,zperig,demiax,excorb,xincli,gomega
      common / oricom / alfpre,gitpre
      common / oritem / datini
      common / parkin / zapotf,zpertf
      common / period / tnavig,tguida,tpilot,tpredi,tinteg
      common / phagui / vsorti
      common / planet / xomega(3),requat,rpolar,natpla
      common / profro / xaltro(5),xgabro(5,2)
      common / revers / irevrs
      common / satorb / enrmin
      common / secgui / iseccp,isecex
      common / succes / errinc,errvit,errzap,errzpe
      common / tabaer / tabmac(nmachx),tabcxe(nmachx),tabcze(nmachx)
      common / tablat / nbalti
      common / tablar / nbmach
      common / tkodak / tphoto
      common / trigon / degrad,pi
      common / xvrent / positz(3),vitesz(3)
      common / vlimit / epsiln
      common / raynez / cq
      common / traref / irefer
      common / gitrfr / gitref
      common / secpdy / pdymax
      common / carpil / cstpil,amrpil,omgpil
      common / modpil / natpil
           
      common / nbgite / ntabll
      common / nbgain / ntabul
      common / tabatm / altatm(1500),romatm(1500)
      common / nbzatm / naltit
      common / capexi / cinexi(6)
      common / colnrj / nbrnrj
      common / tabnrj / nrjval(itable), nrjpre(itable),nrjhdt(itable),
     +                  nrjtst(itable)
      common / reftab / refpre(itable), refhdt(itable),refhtt(itable),
     +                  refincli(itable),refdates(itable),refcmu(itable)
      
      common / gainmu / amorft,pulsft
      common / modpdn / nzapd
      common / varpdn / altpdn(nzapdmx),tabpda(nzapdmx),tabpdb(nzapdmx)
      
      common / varhor / atmvar,ampli,wavlen
      common / varver / atmver,atmdis
      common / recher / pente
c
      intrinsic  datan,dcos,dsin,dsqrt
c
c		initialisation de constantes
c
      pi     = 4.d0*datan(1.d0)
      g0terr = 9.81 d0
      g0mars = 0.d0             
      degrad = pi/180.d0
      epsiln = 1.d-7
      enrmin = 1.d2
c
      gitref = gitref*degrad
c      
      if (natpla.eq.2) then
         rmoyen    = 6.0518   d6         
 	 requat    = 6.0518   d6
	 rpolar    = 6.0518   d6
	 xj2       = 4.458    d-6
	 xmug      = 3.249    d14
         xomega(3) = 2.9924   d-7
      endif
      if (natpla.eq.3) then
         rmoyen    = 6.378137   d6         
 	 requat    = 6.378137   d6          
	 rpolar    = 6.356784   d6
	 xj2       = 1.08263    d-3
	 xmug      = 3.98600418 d14
         xomega(3) = 7.292115 d-5
      endif
c
c			cas Maxtom
c
c      if (natpla.eq.3) then
cc         rmoyen    = 6.370000   d6         
cc 	 requat    = 6.370000   d6          
cc	 rpolar    = 6.370000   d6
c         rmoyen    = 6.378135  d6         
c 	 requat    = 6.378137   d6          
c	 rpolar    = 6.3567523   d6
c	 xj2       = 1.08263    d-3
c	 xmug      = 3.986005 d14
c        xomega(3) = 7.2921151467 d-5
c      endif
      if (natpla.eq.4) then
	 rmoyen    = 3.393940 d6         
	 requat    = 3.393940 d6         
	 rpolar    = 3.376780 d6
	 xj2       = 1.958616 d-3        
	 xmug      = 4.282829 d13
         xomega(3) = 7.088218 d-5
      endif
      if (natpla.eq.5) then
         rmoyen    = 71.492   d6
	 requat    = 71.492   d6
	 rpolar    = 66.854   d6
	 xj2       = 14.736   d-3
	 xmug      = 1.26686  d17
         xomega(3) = 1.759 d-4
      endif		
c
      excent = dsqrt(requat**2 - rpolar**2)/requat
c
      xomega(1) = 0.d0
      xomega(2) = 0.d0
c
c		erreurs finales admissibles visees
c
      read(111,*)
      read(111,*)
      read(111,*)
      read(111,*) errinc
      read(111,*) errvit
      read(111,*) errzap
      read(111,*) errzpe
c
c		lecture des caracteristiques du guidage
c
      read(109,*)
      read(109,*)
      read(109,*)
      read(109,*) amorft
      read(109,*) pulsft
      read(109,*) margmu(1)
      read(109,*) amorth
      read(109,*) pulsah
      read(109,*) vsorti
      read(109,*) margmu(2)
      read(109,*) altcst
      read(109,*) gaindh
      read(109,*) dzalim
      read(109,*) coridx
      read(109,*) coridy
      read(109,*) irevrs
      read(109,*) iseccp
      read(109,*) isecex
      read(109,*) lambda
      read(109,*) pdacti
      read(109,*) pdinib
      read(109,*) enrlat(1)
      read(109,*) enrlat(2)
      read(109,*) pdymax
      read(109,*) nzapd

      if (nzapd.gt.nzapdmx) then
         nzapd = nzapdmx
      endif

      do  i = 1,nzapd
          read(109,*) altpdn(i),tabpda(i)

          tabpdb(i) = 0.d0
      end do

      pulsah    = pulsah*degrad
      coridx    = coridx*1.d0
      coridy    = coridy*degrad
      altcst    = altcst*1.d3
      if (natman.eq.1) then
         pdacti    = pdacti*1.d6
         pdinib    = pdinib*1.d6
         enrlat(1) = enrlat(1)*1.d6
         enrlat(2) = enrlat(2)*1.d6
      endif
c
c		lecture du profil de commande sur trajectoire optimale
c
      if ((atmvar.eq.2).and.(atmver.eq.2)) then
      	irefer = 1
      endif
      
      if (irefer.eq.0) then
         do  i = 1,100000
             read(113,*,end=999) nrjval(i),refpre(i),refhdt(i),
     +                           refhtt(i),refincli(i),refdates(i),
     +                           refcmu(i)

             nrjval(i) = nrjval(i)*1.d6
             refpre(i) = refpre(i)
             refhdt(i) = refhdt(i)

             ntabul = i

         end do

 999     continue
         
      else
         ntabul = 0
      endif
      nbrnrj = ntabul
c
c		lecture du profil de commande en incidence
c
      read(110,*)
      read(110,*)
      read(110,*)
      read(110,*) nbalfa
      do  i = 1,nbalfa
          read(110,*) profax(i)
      end do
      do  i = 1,nbalfa
          read(110,*) profay(i)
      end do
c 
      do  i = 1,nbalfa
          profax(i) = profax(i)*1.d3
      end do
      do  i = 1,nbalfa
          profay(i) = profay(i)*degrad
      end do
c
c		lecture des caracteristiques capsule
c
      read(100,*)
      read(100,*)
      read(100,*)
      read(100,*) xmasse
      read(100,*) srefer
      read(100,*) cq
      read(100,*) vgitmx
      read(100,*) tnavig
      read(100,*) tguida
      read(100,*) tpilot
      read(100,*) tpredi
      read(100,*) tinteg
      read(100,*) tphoto
c
      vgitmx = vgitmx*degrad
c
c		lecture des caracteristiques pilote
c
      read(115,*)
      read(115,*)
      read(115,*)
      read(115,*)  natpil
      read(115,*)  cstpil
      read(115,*)  amrpil
      read(115,*)  omgpil
c
c		lecture des caracteristiques mission
c
      read(101,*)
      read(101,*)
      read(101,*)
      read(101,*) ivents
      read(101,*) conflu
      read(101,*) conacc
      read(101,*) conpdy
      read(101,*) xaltfn
      read(101,*) xlonfn
      read(101,*) xlatfn
      read(101,*) xvitfn
      read(101,*) xpenfn
      read(101,*) xazmfn
      read(101,*) enrjfn
      read(101,*) vitzfn
      read(101,*) zapoge
      read(101,*) zperig
      read(101,*) demiax
      read(101,*) excorb
      read(101,*) xincli
      read(101,*) gomega
      read(101,*) zapotf
      read(101,*) zpertf
      
      if (natman.eq.2) then
         read(101,*) vitinf
         read(101,*) anoinf
         anoinf = anoinf*degrad
      else
         vitinf = 1.d31
         anoinf = 1.d31
      endif
c
      xaltfn = xaltfn*1.d3
      xlonfn = xlonfn*degrad
      xlatfn = xlatfn*degrad
      xpenfn = xpenfn*degrad
      xazmfn = xazmfn*degrad
      demiax = demiax*1.d3
      xincli = xincli*degrad
      gomega = gomega*degrad
      zapoge = zapoge*1.d3
      zperig = zperig*1.d3
      zapotf = zapotf*1.d3
      zpertf = zpertf*1.d3
      enrjfn = enrjfn*1.d6
c
      conacc = conacc*g0terr
      conflu = conflu*1.d3
      conpdy = conpdy*1.d3
c
c		lecture des conditions en debut d'aerocapture
c
      read(102,*)
      read(102,*)
      read(102,*)
      read(102,*) xaltzd
      read(102,*) xlonzd
      read(102,*) xlatzd
      if (irefer.ne.-1) then
         read(102,*) xvitzd
         if (atmvar.eq.2) then
         	read(102,*)
         	xpenzd = pente
         else
                read(102,*) xpenzd
         endif
         read(102,*) xazmzd
      else
c
c		cas de trajectoire de reference en AGA avec Vinf...
c 
         read(102,*) xvitinf
         read(102,*) xaltzp
         read(102,*) xazmabs      
      endif
      read(102,*) datini
      read(102,*) gitpre
      read(102,*) alfpre

      dcx = 0.
      dcz = 0.
      dro = 0.
c     
      xmasse = xmasse
      xaltzd = xaltzd*1.d3
      xlonzd = (xlonzd+dlong)*degrad
      xlatzd = (xlatzd+dlat)*degrad
      
      if (irefer.eq.-1) then
         irefer = 1
         xaltzp = xaltzp*1.d3
         xazmabs = xazmabs*degrad
         xrp = xaltzp + requat
         xr  = xaltzd + requat
         ex  = 1.d0 + xvitinf**2*xrp/xmug
         nu  =-dacos((xrp*(1.d0 + ex)/xr-1.d0)/ex)
         p   = datan2((ex*dsin(nu)),(1.d0+ex*dcos(nu)))
         v = dsqrt(2.d0*xmug/xr+xvitinf**2)
         vx = v*dcos(p)*dsin(xazmabs)
         vy = v*dcos(p)*dcos(xazmabs)
         vz = v*dsin(p)

         vpxy = dsqrt(vy**2+(vx-xomega(3)*xr*dcos(xlonzd))**2)

         xvitzd = dsqrt(vz**2+vpxy**2)
         xazmzd = datan2((vx-xomega(3)*xr*dcos(xlonzd))/vpxy,vy/vpxy)
         xpenzd = datan2(vz/xvitzd,vpxy/xvitzd) 
         
         xazmzd =  xazmzd/degrad
         xpenzd =  xpenzd/degrad     
      endif
      
      xvitzd = (xvitzd+dvit)
      xpenzd = (xpenzd+ks*dpen)*degrad
      xazmzd = (xazmzd+dazm)*degrad
      gitpre = gitpre*degrad
      alfpre = alfpre*degrad
      
      gitpre = gitref
c
c		lecture des conditions en debut de phase de sortie
c
c      read(114,*)
c      read(114,*)
c      read(114,*)
c      read(114,*) cinexi(1)
c      read(114,*) cinexi(2)
c      read(114,*) cinexi(3)
c      read(114,*) cinexi(4)
c      read(114,*) cinexi(5)
c      read(114,*) cinexi(6)
c
      cinexi(1) = cinexi(1)*1.d3
      cinexi(2) = cinexi(2)*degrad
      cinexi(3) = cinexi(3)*degrad
      cinexi(5) = cinexi(5)*degrad
      cinexi(6) = cinexi(6)*degrad
c
c		lecture des tables aerodynamiques
c
      read(104,*)
      read(104,*)
      read(104,*)
      read(104,*) alfaeq
      read(104,*) nbmach
      do  i = 1,nbmach
          read(104,*) tabmac(i),tabcae(i),tabcne(i)
      end do
c
      alfaeq = alfaeq*degrad
      do  i = 1,nbmach
          tabmac(i) = tabmac(i)*degrad
          tabcxe(i) = ((tabcae(i)*dcos(tabmac(i)) + 
     +                  tabcne(i)*dsin(tabmac(i))))*(1. + dcx)
          tabcze(i) =(-tabcae(i)*dsin(tabmac(i)) + 
     +                 tabcne(i)*dcos(tabmac(i)))*(1+dcz)
          
          write(6,*) tabmac(i)/degrad,tabcxe(i),tabcze(i)
      end do
c
      kintcx = 2
      kintcz = 2
c
c		coefs. aeros incidence equilibree, axes engin
c
      call  intrmo (alfaeq,tabmac,tabcxe,nbmach,
     +              kintcx,
     +              cxenom)
      call  intrmo (alfaeq,tabmac,tabcze,nbmach,
     +              kintcz,
     +              czenom)

      finnom      = czenom/cxenom
      ballistique = 1./(xmasse/(srefer*cxenom))
c
c		lecture des tables atmospheriques
c      
      read(105,*)
      read(105,*)
      read(105,*)
      read(105,*) naltit
      do  i = 1,naltit
          read(105,*) altatm(i),romatm(i)
          romatm(i) = romatm(i)*(1. + dro)
      end do
      do  i = 1,5
          read(105,*)  xaltro(i)
      end do
      do  i = 1,5
          read(105,*)  xdisro(i)
      end do 
      read(105,*) rozmod
      read(105,*) facech
      read(105,*) zromod
      read(105,*) cstgam
      
      rozmod = rozmod*(1. + dro)
c
      do  i = 1,5
          xaltro(i)   = xaltro(i)*1.d3
          xdisro(i)   = xdisro(i)/1.d2
          xgabro(i,1) = 0.d0
          xgabro(i,2) = 0.d0
      end do
c
      do  i = 2,5
          xgabro(i,1) =(xdisro(i) - xdisro(i-1))/
     +                 (xaltro(i) - xaltro(i-1))
          xgabro(i,2) = xdisro(i) - xgabro(i,1)*xaltro(i) 
      end do

c
c		affichage des conditions simulations
c
      write(*,2001) xaltzd/1000
      write(*,2002) xlonzd/degrad
      write(*,2003) xlatzd/degrad
      write(*,2004) xvitzd 
      write(*,2005) xpenzd/degrad
      write(*,2006) xazmzd/degrad
      write(*,2007) finnom, (1+dcz/(1+dcx))*100
      write(*,2008) ballistique, (1+dcx)*100
      write(*,2009) romatm(1), (1+dro)*100
      write(*,2010) ks  
c
c		lecture des caracteristiques des dispersions
c
      read(106,*)
      read(106,*)
      read(106,*)
      read(106,*) daltzd
      read(106,*) dlonzd
      read(106,*) dlatzd
      read(106,*) dvitzd
      read(106,*) dpenzd
      read(106,*) dazmzd
      read(106,*) dxdrag
      read(106,*) dxlift
      if (atmver.eq.1) then
      	read(106,*)
      	droatm = atmdis
      else
        read(106,*) droatm
      endif
      read(106,*) dincid
      read(106,*) dmasse
c
      daltzd = xmulti(2)*daltzd*1.d3
      dlonzd = xmulti(2)*dlonzd*degrad
      dlatzd = xmulti(2)*dlatzd*degrad
      dpenzd = xmulti(2)*dpenzd*degrad
      dazmzd = xmulti(2)*dazmzd*degrad
      dxdrag = xmulti(4)*dxdrag
      dxlift = xmulti(4)*dxlift
      dincid = xmulti(4)*dincid*degrad
c
      dxdrag = dxdrag/100.d0
      dxlift = dxlift/100.d0
      droatm = droatm/100.d0
      dmasse = dmasse/100.d0
c
c		lecture des performances de navigation
c
      read(107,*)
      read(107,*)
      read(107,*)
      read(107,*) dnaval
      read(107,*) dnavla
      read(107,*) dnavlo
      read(107,*) dnavvi
      read(107,*) dnavpe
      read(107,*) dnavaz
      read(107,*) dnavad
c
      dnaval = xmulti(1)*dnaval*1.d3
      dnavlo = xmulti(1)*dnavlo*degrad
      dnavla = xmulti(1)*dnavla*degrad
      dnavpe = xmulti(1)*dnavpe*degrad
      dnavaz = xmulti(1)*dnavaz*degrad
      dnavad = xmulti(3)*dnavad
c
c		fermeture des fichiers de donnees
c
      close(unit= 100)
      close(unit= 101)
      close(unit= 102)
      close(unit= 103)
      close(unit= 104)
      close(unit= 105)
      close(unit= 106)
      close(unit= 107)
      close(unit= 109)
      close(unit= 110)
      close(unit= 111)
      close(unit= 115)
cc      close(unit= 114)
      if (irefer.eq.0) then
         close(unit= 113)
      endif
c
c		calculs preliminaires (position, vitesse, ...)
c
      call  geodes (xaltzd,xlatzd,xlonzd,
     +              positz)
c
      vitesz(1) = xvitzd
      vitesz(2) = xpenzd
      vitesz(3) = xazmzd
c
      if ((nbalti.gt.naltix).or.(nbmach.gt.nmachx)) then
         write(6,*)
         write(6,*) 'Dimensions non compatibles des donnees'
         write(6,*)
         write(6,*) 'fichiers de donnees et include a verifier...'
         write(6,*)
         write(6,1000) nbalti,naltix
         write(6,1010) nbmach,nmachx
         write(6,*)
         iconfd = 0
      else
         iconfd = 1
      endif
c
 1000 format(1x,'nombre de points d''altitude tables atmosphere',
     +           2(i3,1x))
 1010 format(1x,'nombre de points de Mach tablea aerodynamiques',
     +           2(i3,1x))

 2001 format(1x,'Altitude.......', f12.6)
 2002 format(1x,'Longitude......', f12.6)
 2003 format(1x,'Latitude.......', f12.6)
 2004 format(1x,'Vitesse........', f12.6)
 2005 format(1x,'Pente..........', f12.6)
 2006 format(1x,'Azimuth........', f12.6)
 2007 format(1x,'Finesse........', f12.6,' (',F6.2,'% valeur nominale)')
 2008 format(1x,'C.ballistique..', f12.6,' (',F6.2,'% valeur nominale)')
 2009 format(1x,'Densite au sol.', f12.6,' (',F6.2,'% valeur nominale)')
 2010 format(1x,'K dispersion...', f12.6)

      return
      end
