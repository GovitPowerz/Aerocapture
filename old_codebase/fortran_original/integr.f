      subroutine  integr (gitraf,nrjfin,positr,vitesr,ecamoy)

      implicit none
            
      integer  ibounc,icaptr,idebut,ifina2,iphase,
     +         iprepr(2),irebon,isecur,nbroll,itera,
     +         i,kintal,kintgu(2),kintnv(2),kinttr(2),nbalfa,nbmach,
     +         indrol,indext,isorti,natsim,
     +         kintop,kintat,kintlp,ilater,iguida(2),
     +         icrash,imodel,incrar,incrat,intrnv(2),
     +         ilongi,isatur,natman,iecran,ncarlo,
     +         imodel,ivents,j,k,kinttr,ix,itable,ntabul
     
      parameter (itable = 2000)

      double precision  xorbit(13),ecartr(4),positr(3),vitesr(3),
     +                  positn(3),vitesn(3),altmax(3),datmax(3),
     +                  fluter(2),fcharg(2),pdynam(2),alfcom,coefro,
     +                  gitpre,sgngit,somflu,somgit,temsim,trebon,
     +                  vitpre,zrebon,nrjfin,
     +                  alfini,datini,
     +                  degrad,demiax,dxdrag,dxlift,
     +                  excorb,gitini,gomega,pi,vitref,
     +                  xaltfn,xazmfn,xincli,xlatfn,xlonfn,
     +                  xpenfn,xvitfn,zapoge,zperig,gitref,
     +                  tlater,dtroll,gpilpr,gitpil,disatm,
     +                  acceln(2),coefan(2),pdynan,
     +                  roguid,tcaptr,vitref,acdrag,
     +                  aclift,altcst,altitu,cstgam,dvabrl,
     +                  dvitrd,dzalim,facech,gaindh,lambda,posita(3),
     +                  rmoyen,rorefr,rozmod,srefer,
     +                  vgitmx,vitabs,vitesa(3),vitrel,vitrad,
     +                  vittot,vphase,xcharg,xflutr,xlatit,
     +                  xmasse,zromod,tnavig,tguida,
     +                  tpilot,tpredi,tinteg,xlongi,g0terr,
     +                  pnorme,coefrp,vitson,xsauve(24),
     +                  sgngit,somgit,vitref,gitcom,vitgit,
     +                  enrjlt,enrlat,pdacti,pdinib,gitref,
     +                  enrtot,excent,xj2,xmug,hpp,acgrav,
     +                  pdyneq,gpilpr,gitpil,alfpil,vitpil,
     +                  accelr(2),altitr,energr,finess,romver,trebon,
     +                  vitmac,vitszr,romnom,alfaeq,coefar,cosazm,
     +                  cosgit,coslat,cospen,cstgam,cxcaps,cxenom,
     +                  czcaps,czenom,dazven,deltat,
     +                  disatm,dviven,facech,finnom,
     +                  gravtl,gravtr,parvit(3,3),rayvec,requat,
     +                  rpolar,sinazm,singit,sinlat,sinpen,
     +                  tanlat,tanpen,temrel,vitaer,vitrel,
     +                  vitson,vittot,cq,
     +                  xaltit,xaltro,xcharg,xderiv(8),xetats(8),
     +                  xincli,xflutr,xgabro,xlatit,xlongi,xmasse,
     +                  xomega,xpdyna,xqk(8),zapoge,zperig,zromod,
     +                  rayvec,latitu,longit,penvit,azmvit,vitess,
     +                  gitraf,nrjval,refpre,refhdt,nrjtst,
     +                  refhtt,refincli,refdates,nrjpre,nrjhdt,
     +                  xinccr,vitcom,tramax,
     +                  refcmu,ecamoy(3),dalfae,memoir(4)
     
      common / orbvis / zapoge,zperig,demiax,excorb,xincli,gomega
      common / period / tnavig,tguida,tpilot,tpredi,tinteg
      common / missio / xaltfn,xlonfn,xlatfn,xvitfn,xpenfn,xazmfn
      common / mecaer / dalfae,disatm,dxdrag,dxlift
      common / modatm / cstgam,facech,rozmod,rmoyen,zromod
      common / oritem / datini
      common / tablar / nbmach
      common / trigon / degrad,pi
      common / oricom / alfini,gitini
      common / capsul / srefer,vgitmx,xmasse
      common / carext / altcst,dzalim,gaindh
      common / estiro / lambda
      common / gravit / g0terr
      common / congui / pdacti,pdinib
      common / geoide / excent,xj2,xmug
      common / loglat / enrlat(2)
      common / modaga / natman
      common / modgui / natsim
      common / planet / xomega(3),requat,rpolar
      common / profro / xaltro(5),xgabro(5,2)
      common / raynez / cq
      common / modven / ivents
      common / aernom / cxenom,czenom,finnom
      common / aeroeq / alfaeq
      common / nbgain / ntabul
      common / tabnrj / nrjval(itable), nrjpre(itable),nrjhdt(itable),
     +                  nrjtst(itable)
      common / reftab / refpre(itable), refhdt(itable),refhtt(itable),
     +                  refincli(itable),refdates(itable),refcmu(itable)
           
      external  enrtot
      
      open(unit=853,file='../sorties/traj_ref2',form='formatted')  

      open(unit=835,file='../sorties/divers.inter2',form='formatted')
      
c 110  continue
      
      gitraf=gitraf*degrad
      
      gitref = gitraf
c
      ncarlo = 0
c
c		erreurs aerodynamiques et atmospheriques
c
      memoir(1)=dalfae
      memoir(2)=disatm
      memoir(3)=dxdrag
      memoir(4)=dxlift
      
      dalfae = 0.
      disatm = ecamoy(1)/100
c      disatm = 0.
      dxdrag = 0.
      dxlift = 0.
c
c		etat estime (prises en compte erreurs de navigation)
c
      do  i = 1,3
          positn(i) = positr(i)
          vitesn(i) = vitesr(i)
      end do
c
c		mises a zero, initialisations diverese
c
      do i = 1,3
         altmax(i) = 0.d0
         datmax(i) = 0.d0
      end do
      do  i = 1,2
          fluter(i) = 0.d0
          fcharg(i) = 0.d0
          pdynam(i) = 0.d0
      end do
c
      coefro = 1.d0
      alfcom = alfini
      gitpre = gitraf
      gpilpr = gitpre
      gitpil = gpilpr
      if (gitraf.le.0.d0) then
         sgngit =-1.d0
      else
         sgngit = 1.d0
      endif
      somflu = 0.d0
      somgit = 0.d0
      temsim = datini
      trebon = 1.d30
      zrebon = 1.d34
      vitpre =-1.d30
      vitref = vitesr(1)*dsin(vitesr(2))
      tlater = 0.d0
      dtroll =-1.d30 
c
      ilater = 0
      ibounc = 0
      icaptr = 0
      idebut = 1
      ifina2 = 0
      indext =-1
      indrol = 0
      iphase = 1
      irebon = 0
      isecur = 0
      isorti = 0
      nbroll = 0
      itera  = 0
      
c
      kintop = 2
      kintlp = 2
      kintat = 50
      kintal = 1 + nbalfa/2
      do  i = 1,2
          kintgu(i) = 1 + nbmach/2
          kintnv(i) = 1 + nbmach/2
          kinttr(i) = 1 + nbmach/2
          iprepr(i) = 0
      end do
c
c		parametres orbitaux reels initiaux
c
      call  orbito (positr,vitesr,
     +              xorbit)
      ecartr(1) = xorbit(1) - demiax
      ecartr(2) = xorbit(2) - excorb
      ecartr(3) = xorbit(3) - xincli
      ecartr(4) = xorbit(4) - gomega
c
c		edition ecran des conditions initiales
c
c      call  etaini (positr,vitesr,isimul)
c      
      do while (ifina2.eq.0) 
      
c
c	navigation
c
      indext = 0
      icrash = 0
c
c		addition des erreurs de navigation (biais constants)
c
      do  i = 1,3
          positn(i) = positr(i)
          vitesn(i) = vitesr(i)
      end do
c
c		vitesse absolue
c
      call  xvabsl (positn,vitesn,
     +              posita,vitesa)
c
      vitabs = pnorme (vitesa)
      vitrel = pnorme (vitesn)
      dvabrl = vitabs - vitrel
      xlongi = 0.d0
c
c		parametres aerodynamiques estimes
c
      incrar = intrnv(1)
      incrat = intrnv(2)
c
c		densite atmospherique (modele embarque guidage)
c  
      call  frayon (positn,
     +              altitu,xlatit)
c
      call  fatmos (altitu,xlatit,xlongi,temsim,imodel,
     +              kintat,
     +              rorefr,vitson)


      coefro = 1.d0
      
      coefrp = coefro
      roguid = coefro*rorefr*(1 + disatm)

      pdynan = 0.5*roguid*(vitesn(1)**2)

c
c		test de rebond
c
      if (ibounc.eq.0) then
         if (dsin(vitesn(2)).gt.0.d0) then
            ibounc = 1
         endif
      endif
c
      vitrad = vitesn(1)*dsin(vitesn(2))
c
c		gestion des phases du guidage longi.
c
      if (ibounc.eq.0) then
c
c		guidage en phase de capture
c
          iphase = 1
      else

         if ((vitesn(1).ge.vphase).and.(vitrad.lt.0)) then
            iphase = 1
         endif
         if ((vitesn(1).le.vphase).and.(iphase.eq.1)) then
c
c		guidage en phase de sortie
c
            iphase = 2
            tcaptr = temsim
            indext = 1
            vitref = vitrad
c        
         endif
      endif
c
c		test de decroissance de dh/dt apres rebond
c
      if (ibounc.ge.1) then
         vitrad = vitesn(1)*dsin(vitesn(2))
         dvitrd = vitrad - vitpre
         vitpre = vitrad
         if (dvitrd.lt.0.d0) then
            icrash = 1
         else
            icrash = 0
         endif
      endif
c
c		securite en cas de capture apres rebond
c
      if (icrash.eq.1) then
         iphase = 3
      else
         if (vitrad.ge.120) then
            iphase = 2
         endif
      endif
c
      iphase=1
      if (iphase.eq.1) then
         tcaptr = tcaptr + tnavig
      endif
c
c		consigne de guidage en incidence
c
      call  guialf (positn,vitesn,roguid,
     +              alfcom)
c
c
c	sauvegarde traj de ref
c
      ilongi    = 0
      iguida(1) = 0
      iguida(2) = 0

      vitrel = vitesn(1)
      vitrad = vitesn(1)*dsin(vitesn(2))
      pdyneq = 0.5d0*roguid*vitrel**2
      acgrav = xmug/positn(1)**2
         
      enrjlt = enrtot (positn,vitesn)     
      
      hpp = dcos(gitref)*srefer*coefan(2)*pdyneq/xmasse -
     +        (acgrav - (vitrel**2/positn(1)))*dcos(vitesn(2))
      
      call  orbito (positn,vitesn,
     +                 xorbit)
      call  frayon (positn,
     +                 altitu,xlatit)      
      xinccr = xorbit(3)
      
      write(853,777) enrjlt/1.d6,pdyneq,vitrad,hpp,xinccr/degrad,
     +                 temsim,dcos(gitref) 
     
 777  format(7(1x,1pe23.16))
      
      gitcom=gitraf
      
c
c	pilotage
c
      alfpil = alfcom
      gitpil = gitcom
      vitpil = vitcom
      
c
c	integration de la traj
c
c
c		initialisations
c
      dviven = 0.d0
      dazven = 0.d0
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
          dxdrag = 0.
          dxlift = 0.

          romver = romver*(1.d0 + disatm)

c
c		vitesse aerodynamique et changement de reperes
c
             vitaer = vitrel
             do  i = 1,3
                 do  j = 1,3
                     parvit(i,j) = 0.d0
                 end do
                 parvit(i,i) = 1.d0
             end do
c
c		expression accelerations aeros en repere vitesse
c
          coefar = romver*srefer/(2.d0*xmasse)
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
      romver = romnom
      vitmac = vitesr(1)/vitson
      
c
      call  conph2 (positr,vitesr,alfpil,temrel,imodel,
     +              incrar,incrat,
     +              coefan,xcharg,xflutr,xpdyna,acdrag,aclift)
c
      
      fluter(1) = xflutr
      fcharg(1) = xcharg
      pdynam(1) = xpdyna
      finess    = czenom/cxenom
     
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
c	test de fin de mission
c
      tramax = 5000.
c
c		test sur altitude negative (crash capsule)
c
	iecran = 0
      if (altitr.le.0.d0) then
         ifina2 = 1
         if (iecran.eq.1) write(6,1000) temsim
      endif
c
c		test sur depassement de temps
c
      if (temsim.ge.tramax) then
         ifina2 = 2
          if (iecran.eq.1) write(6,2000) altitr/1.d3
      endif
c
c		test sur sortie d'atmosphere
c
      if ((irebon.eq.1).and.(altitr.ge.xaltfn)) then
         ifina2 = 3
          if (iecran.eq.1) write(6,3000) temsim
      endif
c
 1000 format(1x,'Arret sur crash Orbiter a T = ',f8.3,' s')
 2000 format(1x,'Arret sur critere temporel a Z = ',f8.3,' km')
 3000 format(1x,'Arret sur altitude fin mission a T = ',f8.3,' s')
 4000 format(1x,'Arret sur changement de phase a T = ',f8.3,' s')
c

c
c	sauvegarde des valeurs intermediaires
c

c
c		parametres reels
c
      rayvec = positr(1)
      longit = positr(2)
      latitu = positr(3)
      vitess = vitesr(1)
      penvit = vitesr(2)
      azmvit = vitesr(3)
c
c		determination altitude geodesique
c
      call  frayon (positr,
     +              altitu,latitu)
c
c		parametres energetiques
c
      vitrad = vitess*dsin(penvit)
c
      xsauve(1)  = temsim
      xsauve(2)  = fluter(1)/1.d3
      xsauve(3)  = fcharg(1)/g0terr
      xsauve(4)  = pdynam(1)/1.d3
      xsauve(5)  = accelr(1)/g0terr
      xsauve(6)  = accelr(2)/g0terr
      xsauve(7)  = gitcom/degrad
      xsauve(8)  = gitpil/degrad
      xsauve(9)  = vitgit/degrad
      xsauve(10) = vitmac
      xsauve(11) = acceln(1)/g0terr
      xsauve(12) = acceln(2)/g0terr
      xsauve(13) = romver
      xsauve(14) = somflu/1.d6
      xsauve(15) = vitrad
      xsauve(16) = isatur
      xsauve(17) = roguid
      xsauve(18) = coefro
      xsauve(19) = isecur
      xsauve(20) = altitu/1.d3
      xsauve(21) = vitrad
      xsauve(22) = energr/1.d6
      xsauve(23) = alfcom/degrad
      xsauve(24) = alfpil/degrad
c
      write(835,1002) (xsauve(k), k = 1,24)      
 1002 format(24(1x,d20.10))

      temsim = temsim + 1  

      end do
      close(835)
      close(853)
      
      nrjfin = energr/1.d6
      
      dalfae = memoir(1)
      disatm = memoir(2)
      dxdrag = memoir(3)
      dxlift = memoir(4)
      
      open(unit= 955, file= '../sorties/traj_ref2',
     +                   form= 'formatted')

      do  i = 1,100000
          read(955,*,end=999) nrjval(i),refpre(i),refhdt(i),
     +                        refhtt(i),refincli(i),refdates(i),
     +                        refcmu(i)
     
          nrjval(i) = nrjval(i)*1.d6
          refpre(i) = refpre(i)
          refhdt(i) = refhdt(i)
                          
          ntabul = i
             
      end do
               
 999  close(955) 
      
 8532 format(t64,d,310x,d)

      return
      end 

