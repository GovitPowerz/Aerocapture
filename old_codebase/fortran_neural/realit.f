c1
c1    copyright (c) AEROSPATIALE 1999
c1......................................................................
c2    nom    : realit.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine la trajectoir reeellement suivie par la capsu-
c3    durant la phase d'aerocapture a partir de la commande fournie par
c3    le guidage et realisee par le pilotage.
c3    La determination de la trajectoire se fait par integration numeri-
c3    que par la methode de Ruinge-Kutta d'ordre 4.
c3    La condition de fin d'aerocapture est le passage par l' altitude
c3    de 130 km (sortie d'atnmosphere) apres rebond sur les couches den-
c3    ses de l'atmosphere.
c3
c3    NOTA  Position et vitesse sont exprimees en coordonnees spheriques
c3          dans un repere geocentrique (position) ou local (vitesse).
c3          Les dispersions envisagees sur l'aerodynamique et l'atmosphe
c3          re sont supposees constantes sur une simulation (pas de pro-
c3          fil d'evolution particulier).
c3          Actuellement, on ne tient pas compte de vent et de disper-
c3          sion sur le vent (dviven = 0)
c3......................................................................
c4    variables d'entree
c4
c4    alfpil            R8    incidence commandee
c4    gitpil            R8    gite realisee par le pilote
c4    temsim            R8    temps courant
c4......................................................................
c5    variables d'entree-sortie
c5
c5    positr(3)         R8    position reelle repere geocentrique
c5    vitesr(3)         R8    vitesse reelle repere local
c5    altamx(3)         R8    altitude parametres max.
c5    datmax(3)         R8    dates parametres max.
c5    fluter(2)         R8    flux thermique courant et max.
c5    fcharg(2)         R8    facteur de charge courant et max.
c5    pdynam(2)         R8    pression dynamique courante et max.
c5    somflu            R8    integrale de flux
c5    xincid            R8    incidence
c5    irebon            I4    indicateur de rebond
c5......................................................................
c6    variables de sortie
c6
c6    xorbit(7)         R8    parametres orbitaux
c6    ecartr(4)         R8    ecart courant avec les contraintes finales
c6    accelr(2)         R8    accelerations aerodynamqiues
c6    altitr            R8    altitude
c6    energr            R8    energie totale
c6    finess            R8    finesse vehicule
c6    romver            R8    masse volumique de l'air
c6    trebon            R8    date de rebond atmospherique
c6    vitmac            R8    nombre de Mach
c6    vitszr            R8    vitesse radiale
c5    zrebon            R8    altitude de premier rebond
c6......................................................................
c7    variables internes
c7
c7    xaltit            R8    altitude
c7    xderiv(6)         R8    etaat cinematique derive
c7    xetats(6)         R8    etat cinematique
c7    xlatit            R8    latitude
c7......................................................................
c8    composants appelants
c8
c8    simmsr            INT   simualtion aerocapture
c8......................................................................
c9    composants appeles
c9
c9    conphy            INT   parametres physiaues (flux,Pdyn,...)
c9    energi            INT   parametres energetiques
c9    faeros            INT   coefficients aerodynamiques
c9    fatmos            INT   coefficients atmopsheriques
c9    fgravi            INT   composantes gravite
c9    frayon            INT   rayon planete
c9    fvents            INT   caracteristiques vent
c9    orbito            INT   parametres orbitaux
c9    rkutta            INT   integration numerique Runge-Kutta 4
c9......................................................................
c10   commons utilises
c10
c10   aeroeq                  incidence equilibree
c10   capsul                  caracteristiques capsule
c10   intrea                  increments d'interpolation des tables
c10   mecaer                  meconnaissances aeros et atmopsheriques
c10   modatm                  modele d'atmosphere exponentiel
c10   modven                  modelisation du vent
c10   orbvis                  caracteristiques orbite visee
c10   period                  cadences d'integration
c10   planet                  caracteristiques planete
c10   profro                  profil de dispersion de la densite
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   non                     parametres variables common / intrea /
c11.....................................................................
c
      subroutine  realit  (alfpil,gitpil,temsim,
     +                     positr,vitesr,altmax,datmax,fluter,
     +                     fcharg,pdynam,somflu,irebon,
     +                     xorbit,ecartr,accelr,altitr,energr,
     +                     finess,romver,romnom,trebon,vitmac,
     +                     vitszr,zrebon)
c
      implicit none
c
      integer  irebon,
     +         i,imodel,incrar,incrat,indalt,ivents,j,k,kinttr,
     +         ix,atmvar
c
      double precision  alfpil,gitpil,temsim,positr(3),vitesr(3),
     +                  altmax(3),datmax(3),fluter(2),fcharg(2),
     +                  pdynam(2),somflu,zrebon,xorbit(13),ecartr(4),
     +                  accelr(2),altitr,energr,finess,romver,trebon,
     +                  vitmac,vitszr,romnom,
     +                  acdrag,aclift,alfaeq,coefan(2),coefar,cosazm,
     +                  cosgit,coslat,cospen,cstgam,cxcaps,cxenom,
     +                  czcaps,czenom,dadrag,dalfae,dazven,deltat,
     +                  demiax,disatm,dispro,dviven,dnlift,dxdrag,
     +                  dxlift,excorb,facech,finnom,gomega,gravtl,
     +                  gravtr,parvit(3,3),rayvec,requat,rmoyen,rozmod,
     +                  rpolar,sinazm,singit,sinlat,sinpen,srefer,
     +                  tanlat,tanpen,temrel,tinteg,tguida,tnavig,
     +                  tpilot,tpredi,vgitmx,vitaer,vitdir,vitrel,
     +                  vitson,vittot,vitven(3),vventm,vventz,cq,
     +                  xaltit,xaltro,xcharg,xderiv(8),xetats(8),
     +                  xincli,xflutr,xgabro,xlatit,xlongi,xmasse,
     +                  xomega,xpdyna,xqk(8),zapoge,zperig,zromod,
     +                  dxmass,positz(3),vitesz(3),poscur(3),pi,
     +                  ampli,wavlen,y,poscar(3),positc(3),degrad
c
      common / aernom / cxenom,czenom,finnom
      common / aeroeq / alfaeq
      common / capsul / srefer,vgitmx,xmasse
      common / intrea / kinttr(2)
      common / mecaer / dalfae,disatm,dadrag,dnlift
      common / mecmas / dxmass
      common / modatm / cstgam,facech,rozmod,rmoyen,zromod
      common / modven / ivents
      common / orbvis / zapoge,zperig,demiax,excorb,xincli,gomega
      common / period / tnavig,tguida,tpilot,tpredi,tinteg
      common / planet / xomega(3),requat,rpolar
      common / profro / xaltro(5),xgabro(5,2)
      common / raynez / cq
      
      common / xvrent / positz,vitesz
      common / trigon / degrad,pi
      common / varhor / atmvar,ampli,wavlen
c
      intrinsic  dcos,dmax1,dsin,dsqrt
c
c		initialisations
c
c      disatm = -0.74
      dviven = 0.d0
      dazven = 0.d0
c
c		dispersion sur l'incidence d'equilibre
c      
      alfpil = alfpil + dalfae
c
      incrar = kinttr(1)
      incrat = kinttr(2)
      ix     = 0
      imodel = 0
      deltat = tinteg
c
      do  i = 1,3
          xetats(i)   = positr(i)
          xetats(i+3) = vitesr(i)
      end do
      xetats(7) = somflu
      xetats(8) = temsim
      temrel    = temsim
c
c		valeur de dispersion de l'atmosphere (selon gabarit)
c
      call  frayon (positr,
     +              xaltit,xlatit)
      indalt = 0
      dispro = 0.d0
      do  while (indalt.ne.4) 
          indalt = indalt + 1
          if ((xaltit.ge.xaltro(indalt)).and.
     +        (xaltit.lt.xaltro(indalt+1))) then
             dispro = xgabro(indalt+1,1)*xaltit + xgabro(indalt+1,2)
          endif
      end do
c
c		integration par Runge-Kutta 4
c
      do  k = 1,4
c
c		gravite courante
c
          rayvec = xetats(1)
          xlongi = xetats(2)
          xlatit = xetats(3)
          
          call  fgravi (rayvec,xlatit,
     +                  gravtl,gravtr)
c
c		altitude courante
c
          call  frayon (positr,
     +                  xaltit,xlatit)
c
c		coefficients atmospheriques
c
          call  fatmos (xaltit,xlatit,xlongi,temrel,imodel,
     +                  incrat,
     +                  romver,vitson)
c
c		aerodynamique courante
c
          vitrel = xetats(4)
          vitmac = vitrel/vitson
          
          call  faeros (alfpil,
     +                  incrar,
     +                  cxcaps,czcaps)
     
c
c		dispersions aerodynamiques et atmospheriques
c
          dxdrag = dadrag*dcos(alfpil) + dnlift*dsin(alfpil)
          dxlift =-dadrag*dsin(alfpil) + dnlift*dcos(alfpil)

          romver = romver*(1.d0 + dispro*disatm)
          
          if (atmvar.ge.1) then
          call geodes(xaltit,xlatit,xlongi,poscur)
          call cartes(poscur,0,poscar)
          call cartes(positz,0,positc)
          y=(positc(1)-poscar(1))**2+(positc(2)-poscar(2))**2
          y=y+(positc(3)-poscar(3))**2-(poscur(1)-positz(1))**2
          y=sqrt(y)/1000.
          romver=romver*(1+ampli*(dsin(2*pi*y/wavlen+3*pi/4)))
          endif
          
          cxcaps = cxcaps*(1.d0 + dxdrag)
          czcaps = czcaps*(1.d0 + dxlift)
c
c		vitesse aerodynamique et changement de reperes
c
          if (ivents.eq.0) then
c
c			simulation sans vent
c
             vitaer = vitrel
             do  i = 1,3
                 do  j = 1,3
                     parvit(i,j) = 0.d0
                 end do
                 parvit(i,i) = 1.d0
             end do
          else
c
c			simulation avec vent
c
             call  fvents (xaltit,
     +                     vventm,vventz)
c
             vitven(1) = vventz*dviven
             vitven(2) = vventm*dcos(xetats(6) - dazven)
             vitven(3) = vventm*dsin(xetats(6) - dazven)
             cospen = dcos(xetats(5))
             sinpen = dsin(xetats(5))
c
             vitaer = dsqrt(vitrel**2 +
     +                      vitven(1)**2 -
     +                      2.d0*vitrel*vitven(2)*cospen)
             vitdir = dsqrt(vitrel**2*cospen**2 -
     +                      vitven(1)**2 -
     +                      2.d0*vitrel*vitven(2)*cospen)
c
             parvit(1,1) =(vitrel - vitven(2)*cospen)/vitaer
             parvit(2,1) = vitven(2)*sinpen/vitaer
             parvit(3,1) = vitven(3)/vitaer
             parvit(1,2) =(vitven(1)**2*sinpen -
     +                     vitven(2)*vitrel*cospen*sinpen)/
     +                    (vitaer*vitdir)
             parvit(2,2) =(vitven(1)**2*cospen +
     +                     vitrel**2*cospen -
     +                     vitrel*vitven(2)*(1.d0 + cospen**2))/
     +                    (vitaer*vitdir)
             parvit(3,2) =-vitrel*vitven(3)*sinpen/(vitaer*vitdir)
             parvit(1,3) =-vitven(3)*cospen/vitdir
             parvit(2,3) = vitven(3)*sinpen/vitdir
             parvit(3,3) =(vitrel*cospen - vitven(2))/vitdir
c
         endif
c
c		expression accelerations aeros en repere vitesse
c
          coefar = romver*srefer/(2.d0*xmasse*(1.d0 + dxmass))
          acdrag = coefar*cxcaps*vitaer**2
          aclift = coefar*czcaps*vitaer**2
c
c		equations differentielles du mouvement du cdg
c
          cosgit = dcos(gitpil)
          singit = dsin(gitpil)
          cospen = dcos(xetats(5))
          sinpen = dsin(xetats(5))
          cosazm = dcos(xetats(6))
          sinazm = dsin(xetats(6))
          coslat = dcos(xetats(3))
          sinlat = dsin(xetats(3))
          tanpen = sinpen/cospen
          tanlat = sinlat/coslat
c
c		evolution position (altitude, longitude, latitude)
c
          xderiv(1) = vitrel*sinpen
          xderiv(2) = vitrel*cospen*sinazm/
     +                (rayvec*coslat)
          xderiv(3) = vitrel*cospen*cosazm/
     +                rayvec
c
c		evolution vitesse (norme, pente, azimut)
c
          xderiv(4) =-acdrag*parvit(1,1) - gravtr*sinpen -
     +                gravtl*cospen*cosazm +
     +                aclift*(parvit(1,2)*cosgit + parvit(1,3)*singit) +
     +                xomega(3)**2*rayvec*coslat*
     +                (coslat*sinpen - sinlat*cospen*cosazm)
          xderiv(5) =(aclift*(parvit(2,2)*cosgit + parvit(2,3)*singit)/
     +                vitrel) +
     +               (vitrel*cospen/rayvec) -
     +               ((gravtr*cospen - gravtl*sinpen*cosazm)/vitrel) +
     +               (2.d0*xomega(3)*sinazm*coslat) +
     +               (-acdrag*parvit(2,1)/vitrel) +
     +               (xomega(3)**2*rayvec*coslat*
     +                (sinlat*sinpen*cosazm + coslat*cospen)/vitrel)
          xderiv(6) =(aclift*(parvit(3,2)*cosgit + parvit(3,3)*singit)/
     +                (vitrel*cospen)) +
     +               (vitrel*cospen*sinazm*tanlat/rayvec) +
     +               (2.d0*xomega(3)*(sinlat - cosazm*coslat*tanpen)) +
     +               (gravtl*sinazm/(vitrel*cospen)) +
     +               (xomega(3)**2*rayvec*coslat*sinlat*sinazm/
     +                (vitrel*cospen))
c
c		integrale de flux
c
           xderiv(7) = cq*dsqrt(romver)*vitaer**3.05
c
c		temsp courant
c
           xderiv(8) = 1.d0
c
c		integration numerique
c
          call  rkutta (deltat,xderiv,k,8,ix,
     +                  xqk,
     +                  xetats)
c
          do  i = 1,3
              positr(i) = xetats(i)
          end do
          temrel = xetats(8)
      end do
c
c		restitution des resultats
c
      do  i = 1,3
          positr(i) = xetats(i)
          vitesr(i) = xetats(i+3)
      end do
      somflu  = xetats(7)
      temrel = xetats(8)
c
      xlongi = positr(2)
c
c		altitude
c
      call  frayon (positr,
     +              altitr,xlatit)
c
c		valeurs maximales recontrees
c
      call  fatmos (altitr,xlatit,xlongi,temrel,imodel,
     +              incrat,
     +              romnom,vitson)
c
      romver = romnom*(1.d0 + dispro*disatm)
      vitmac = vitesr(1)/vitson
      
      if (atmvar.ge.1) then
      
      call geodes(altitr,xlatit,xlongi,poscur)
      
      call cartes(poscur,0,poscar)
      
      call cartes(positz,0,positc)
      
      y=(positc(1)-poscar(1))**2+(positc(2)-poscar(2))**2
      y=y+(positc(3)-poscar(3))**2-(poscur(1)-positz(1))**2
      y=sqrt(y)/1000.
      
      romver=romver*(1+ampli*(dsin(2*pi*y/wavlen+3*pi/4)))
      	
      endif
      
c
      call  conphy (positr,vitesr,alfpil,temrel,imodel,
     +              incrar,incrat,
     +              coefan,xcharg,xflutr,xpdyna,acdrag,aclift)
c
      
      fluter(1) = xflutr
      fcharg(1) = xcharg
      pdynam(1) = xpdyna
      finess    =(czenom*(1.d0 + dnlift))/
     +           (cxenom*(1.d0 + dadrag))
     
      finess    = czcaps/cxcaps
c
      xflutr = dmax1(fluter(1),fluter(2))
      xcharg = dmax1(fcharg(1),fcharg(2))
      xpdyna = dmax1(pdynam(1),pdynam(2))
c
      if (xflutr.gt.fluter(2)) then
         fluter(2) = xflutr
         altmax(1) = altitr
         datmax(1) = temsim
      endif
      if (xcharg.gt.fcharg(2)) then
         fcharg(2) = xcharg
         altmax(2) = altitr
         datmax(2) = temsim
      endif
      if (xpdyna.gt.pdynam(2)) then
         pdynam(2) = xpdyna
         altmax(3) = altitr
         datmax(3) = temsim
      endif
c
c		parametres energetiques
c
      call  energi (positr,vitesr,
     +              energr,vitszr,vittot)
c
c		parametres orbitaux
c
      call  orbito (positr,vitesr,
     +              xorbit)
c
c		ecarts courants par rapport aux contraintes missions
c
      ecartr(1) = xorbit(1) - demiax
      ecartr(2) = xorbit(2) - excorb
      ecartr(3) = xorbit(3) - xincli
      ecartr(4) = xorbit(4) - gomega
c
c		determination du (1er) rebond sur l'atmosphere
c
      if (irebon.eq.0) then
         if (dsin(vitesr(2)).ge.0.d0) then
            irebon = 1
            zrebon = altitr
            trebon = temsim
         endif
      endif
c
      kinttr(1) = incrar
      kinttr(2) = incrat
c
      accelr(1) = acdrag
      accelr(2) = aclift
c
      return
      end
