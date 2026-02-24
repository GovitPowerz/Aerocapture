c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : etaini.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module edite a l'ecran les conditions de simulation
c3
c3......................................................................
c4    variables d'entree
c4
c4    positr(3)         R8    position absolue geocentrique spherique
c4    vitesr(3)         R8    vitesse relative locale spherique
c4    isimul            I4    numero de simulation
c4......................................................................
c8    composants appelants
c8
c8    inimsr            INT   initialisation aerocapture
c8......................................................................
c10   commons utilises
c10
c10   coefct                  ponderation des contraintes
c10   congui                  criteres d'inhibition du guidage
c10   mecaer                  meconnaissances aero et atmospheriques
c10   missio                  caracteristiques mission
c10   nrjvis                  parametres energetiques vises
c10   orbvis                  caracteristiques orbite visee
c10   ordcon                  ordonnancement des contraintes terminales
c10   parkin                  caracteristiques orbite de parking
c10   perinj                  dispersions a l'injection
c10   pernav                  erreurs de navigation
c10   trigon                  constantes trigonometriques
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  etaini  (positr,vitesr,isimul)
c
      implicit none
c
      integer  isimul,atmvar,atmver,
     +         ivents,iseccp,isecex,natpla,irefer,natman
c
      double precision  positr(3),vitesr(3),
     +                  dalfae,degrad,demiax,disatm,dispos,disvit,
     +                  dxdrag,dxlift,dxposi,dxvite,enrjfn,excorb,
     +                  gomega,pdacti,pdinib,pi,poncon,vitzfn,disacd,
     +                  enrlat,xaltfn,xazmfn,xincli,xlatfn,xlonfn,
     +                  xorbit(13),xpenfn,xvitfn,zapoge,zapotf,zperig,
     +                  zpertf,xomega,requat,rpolar,gitref,xmulti
     
      double precision ampli,wavlen,atmdis
c
      common / coefct / poncon(4)
      common / congui / pdacti,pdinib
      common / loglat / enrlat(2)
      common / mecaer / dalfae,disatm,dxdrag,dxlift
      common / missio / xaltfn,xlonfn,xlatfn,xvitfn,xpenfn,xazmfn
      common / modaga / natman
      common / muldis / xmulti(4)
      common / nrjvis / enrjfn,vitzfn
      common / orbvis / zapoge,zperig,demiax,excorb,xincli,gomega
      common / parkin / zapotf,zpertf
      common / perinj / dxposi(3),dxvite(3)
      common / pernav / dispos(3),disvit(3),disacd
      common / planet / xomega(3),requat,rpolar,natpla
      common / secgui / iseccp,isecex
      common / trigon / degrad,pi
      common / traref / irefer
      
      common / varhor / atmvar,ampli,wavlen
      common / varver / atmver,atmdis
      common / gitrfr / gitref
c
c		parametres orbitaux en fin de simulation
c
      call  orbito (positr,vitesr,
     +              xorbit)
c
c		edition ecran des resultats de simulation
c
      write(6,*)
      write(6,1000) isimul
      write(6,*)
      if (isimul.eq.1) then
         if (natpla.eq.3) then
            write(6,*) 'Aerocapture TSR '
         endif
         if (natpla.eq.4) then
            write(6,*) 'Aerocapture MSR '
         endif
         if (natpla.eq.5) then
            write(6,*) 'Aerocapture ESR '
         endif
         write(6,*)
         if (ivents.eq.0) then
            write(6,1100)
         else
            write(6,1110)
         endif
         if (irefer.eq.1) then
            write(6,1113) gitref/degrad
         endif
         write(6,1114) xmulti(1)
         write(6,1115) xmulti(2)
         write(6,1116) xmulti(3)
         write(6,1117) xmulti(4)         

         write(6,*)
         if (natman.eq.1) then
            write(6,1600) pdacti/1.d6
            write(6,1610) pdinib/1.d6
         else
            write(6,1800) pdacti
            write(6,1810) pdinib         
         endif
         write(6,*)
         write(6,2999)
         write(6,*)
         write(6,3000) enrjfn/1.d6,vitzfn
         write(6,3100) xaltfn/1.d3,xvitfn
         write(6,3110) xlonfn/degrad,xpenfn/degrad
         write(6,3120) xlatfn/degrad,xazmfn/degrad
         write(6,*)
         write(6,3200) zapoge/1.d3,zperig/1.d3
         write(6,3220) demiax/1.d3,gomega/degrad
         write(6,3210) excorb,xincli/degrad
         write(6,*)
         write(6,3215)
         write(6,3205) zapotf/1.d3,zpertf/1.d3
         write(6,*)
         write(6,3207) iseccp
         write(6,3208) isecex
         if (natman.eq.1) then
            write(6,3209) enrlat(1)/1.d6
            write(6,3216) enrlat(2)/1.d6
         else
            write(6,4209) enrlat(1)
            write(6,4216) enrlat(2)         
         endif
         write(6,*)
c
      endif
c
      write(6,*)
      write(6,2000)
      write(6,*)
      write(6,2100) dxposi(1)/1.d3,dxvite(1)
      write(6,2110) dxposi(2)/degrad,dxvite(2)/degrad
      write(6,2120) dxposi(3)/degrad,dxvite(3)/degrad
      write(6,*)
      write(6,2010)
      write(6,*)
      write(6,2100) dispos(1)/1.d3,disvit(1)
      write(6,2110) dispos(2)/degrad,disvit(2)/degrad
      write(6,2120) dispos(3)/degrad,disvit(3)/degrad
      write(6,2130) disacd
      write(6,*)
      write(6,2200) disatm*100.d0
      write(6,2190) dalfae/degrad
      write(6,2300) dxdrag*100.d0,dxlift*100.d0
      write(6,*)
      write(6,2400)
      write(6,*)
      write(6,2410) xorbit(1)/1.d3
      write(6,2420) xorbit(2)
      write(6,2430) xorbit(3)/degrad
      write(6,2440) xorbit(4)/degrad
      write(6,2445) xorbit(5)/degrad
      write(6,2446) xorbit(8)/degrad
      write(6,2450) xorbit(7)/1.d3
      write(6,2460) xorbit(6)/1.d3
      write(6,2461) xorbit(9)
      write(6,2462) xorbit(10)/degrad
      write(6,2463) xorbit(11)
      write(6,2464) xorbit(12)
      write(6,2465) xorbit(13)
      if (atmvar.eq.1) then
      write(6,*)
      write(6,2466) wavlen
      endif
      if (atmver.eq.1) then
      write(6,*)
      write(6,2467) atmdis
      endif
      write(6,*)
      write(6,*)
c      write(6,*) '       Etot ref','          Pdynref',
c     &           '              vitrad ref'
      
c
 1000 format(5x,'Simulation ',i5)
 1100 format(5x,'           sans vent')
 1110 format(5x,'           avec vent')
 1113 format(5x,'trajectoire de reference a gite cste :',f11.3,' deg')
 1114 format(5x,'erreurs de nav. aerocapture         x ',f11.3)
 1115 format(5x,'erreurs de nav interplanetaire      x ',f11.3) 
 1116 format(5x,'erreurs de mesure accelero          x ',f11.3) 
 1117 format(5x,'erreurs de modele aero              x ',f11.3) 
 1800 format(5x,'seuil d''activation du guidage longi :',f11.3,' Pa')
 1810 format(5x,'seuil d''inhibition du guidage longi :',f11.3,' Pa')
 1600 format(5x,'seuil d''activation du guidage longi :',f11.3,
     +         ' MJ/kg')
 1610 format(5x,'seuil d''inhibition du guidage longi :',f11.3,
     +         ' MJ/kg ')
c
 2000 format(1x,'Erreurs debut rentree')
 2010 format(1x,'Erreurs navigation ')
 2100 format(1x,1('altitude  ',f11.3,' km   vitesse ',f11.3,' m/s '))
 2110 format(1x,1('longitude ',f11.3,' deg  pente   ',f11.3,' deg '))
 2120 format(1x,1('latitude  ',f11.3,' deg  azimut  ',f11.3,' deg '))
 2130 format(1x,'trainee   ',f11.3,' m/s2')
 2190 format(1x,'incidence         ',f11.3,' deg')
 2200 format(1x,'atmopshere     ro ',f11.3,' %')
 2300 format(1x,'aerodynamique  Ca ',f11.3,' %   Cn ',f11.3,' %')
 2400 format(1x,'Orbite initiale')
 2410 format(1x,'demi grand axe ',f13.3,' km')
 2420 format(1x,'excentricite   ',f13.3)
 2430 format(1x,'inclinaison    ',f13.3,' deg')
 2440 format(1x,'Omega          ',f13.3,' deg')
 2445 format(1x,'omega          ',f13.3,' deg')
 2446 format(1x,'nu             ',f13.3,' deg')
 2450 format(1x,'Z apoastre     ',f13.3,' km')
 2460 format(1x,'Z periastre    ',f13.3,' km')
 2461 format(1x,'vinfini	',f13.3,' m/s ')
 2462 format(1x,'nuinfini	',f13.3,' deg ')
 2463 format(1x,'vinfx  	',f13.3,' m/s ')
 2464 format(1x,'vinfy  	',f13.3,' m/s ')
 2465 format(1x,'vinfz  	',f13.3,' m/s ')
 2466 format(1x,'wavlen  	',f13.3,' km')
 2467 format(1x,'atmdis  	',f13.3,' %')
c
 2999 format(1x,'Parametres Mission')
 3000 format(1x,'Energie       ',f11.3,' MJ/kg dh/dt       ',f11.3,
     +          ' m/s')
 3100 format(1x,'altitude      ',f11.3,' km    Vitesse     ',f11.3,
     +          ' m/s')
 3110 format(1x,'longitude     ',f11.3,' deg   pente       ',f11.3,
     +          ' deg')
 3120 format(1x,'latitude      ',f11.3,' deg   azimut      ',f11.3,
     +          ' deg')
 3200 format(1x,'Apoastre      ',f11.3,' km    Periastre   ',f11.3,
     +          ' km')
 3220 format(1x,'demigrand axe ',f11.3,' km    Omega       ',f11.3,
     +          ' deg')
 3210 format(1x,'excentricite  ',f11.3,'       inclinaison ',f11.3,
     +          ' deg')
 3215 format(1x,'Orbite de parking')
 3205 format(1x,'Apoastre     ',f11.3,' km    Periastre   ',f11.3,
     +          ' km')
 3207 format(1x,'mode de securisation guidage capture ',i4)
 3208 format(1x,'mode de securisation guidage sortie  ',i4)
 4209 format(1x,'seuil energetique guidage lateral    ',f11.3,' Pa')
 4216 format(1x,'seuil energetique guidage lateral    ',f11.3,' Pa')
 3209 format(1x,'seuil d activation guidage lateral   ',f11.3,' MJ/kg')
 3216 format(1x,'seuil d inhibition guidage lateral   ',f11.3,' MJ/kg')
c
      return
      end
