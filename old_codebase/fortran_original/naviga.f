c1
c1    copyright (c) AEROSPATIALE 1999
c1......................................................................
c2    nom    : naviga.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module realise la navigationd e la capsule pendant la phase d'a
c3    erocapture.
c3    Actuellement, la modelisation adoptee pour la navigation est tres
c3    simplifiee: on realise un tirage de bruits selon un gabarit donne
c3    fonction de l'altitude, et on ajoute les biais ainsi generees a l'
c3    etat reel.
c3    Actuellement, on n'envisage q'un seul tranche d'altitude.
c3
c3    NOTA on rappelle que la position est fournie par le triplet (alti-
c3         tude-longitude-latitude) et la vitesse par le triplet (norme-
c3         pente-azimut).
c3......................................................................
c4    variables d'entree
c4
c4    positr(3)         R8    position reelle repere geocentrique
c4    vitesr(3)         R8    vitesse reelle repere local
c4    alfcom            R8    incidence commandee precedente
c4    temsim            R8    temps courant
c4......................................................................
c5    variables d'entree-sortie
c5
c5    coefro            R8    coefficient d'estimation de la densite
c5    vizpre            R8    vitesse radiale estimee precedente
c5    ibounc            I4    indicateur de rebond
c5    iphase            I4    indicateur de phase du guidage (vol equili
c5                            bre ou phase de sortie)
c5......................................................................
c6    variables de sortie
c6
c6    ecartn(4)         R8    ecart courant avec contraintes finales
c6    positn(3)         R8    position estimee par la navigation
c6    vitesn(3)         R8    vitesse estimee par la navigation
c6    acceln(2)         R8    accelerations aerodynamiques
c6    coefan(2)         R8    coefficients aerodynamiques estimees
c6    energn            R8    energie totale
c6    pdynan            R8    pression dynamique
c6    roexit            R8    densite atmospherique finale predite
c6    roguid            R8    densite atmospherique courante estimee
c6    tcaptr            R8    duree de la phase de capture
c6    vitszn            R8    vitesse radiale
c6    icrash            I4    indicateur de decroissance de dh/dt
c6    indext            I4    indicateur de passage en phase de sortie
c6......................................................................
c7    variables internes
c7
c7    acdrag            R8    acceleration de trainee
c7    aclift            R8    acceleration de portance
c7    acdram            R8    acceleration de trainee mesuree
c7    bdrag		R8    bruit sur la trainee
c7    coefro		R8    coefficient d'estimation de la densite 
c7    lambda		R8    parametre de la boucle de retour du filtre
c7    imodel            I4    nature du modele d'atmosphere utilise
c7    roesti		R8    densite atmospherique estimee
c7    roref		R8    densite atmospherique sans estimateur
c7......................................................................
c8    composants appelants
c8
c8    simmsr            INT   simulation aerocapture
c8......................................................................
c9    composants appeles
c9
c9    conphy           INT    contraintes mecaniques
c9    energi           INT    parametres energetiques
c9......................................................................
c10   commons utilises
c10
c10   capsul                  caracteristiques capsule
c10   gravit                  constantes pesanteur
c10   intnav                  increments tables d'interpolation
c10   orbvis                  caracteristiques orbite visee
c10   pernav                  gabarit erreurs de navigation
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   non                     parametres variables common / intnav /
c11                                                common / intgui /
c11                                                common / corgui /
c11.....................................................................
c
      subroutine  naviga (positr,vitesr,alfcom,temsim,icarlo,
     +                    coefro,vitpre,ibounc,iphase,
     +                    ecartn,positn,vitesn,acceln,coefan,
     +                    energn,pdynan,roexit,roguid,tcaptr,
     +                    vitref,icrash,indext)
c
      implicit none
c
      integer  icarlo,ibounc,iphase,icrash,indext,
     +         i,imodel,incrar,incrat,intrgu,intrnv,kintat,kintaa
c
      double precision  positr(3),vitesr(3),alfcom,temsim,coefro,
     +                  vitpre,ecartn(4),positn(3),vitesn(3),
     +                  acceln(2),coefan(2),energn,pdynan,roexit,
     +                  roguid,tcaptr,vitref,
     +                  acdrag,acdram,aclift,altcst,altitu,cstgam,
     +                  demiax,disdra,dispos,disvit,dvabrl,dvitrd,
     +                  dzalim,excorb,facech,gaindh,gomega,lambda,
     +                  posita(3),rmoyen,roesti,rorefr,rorfex,rozmod,
     +                  srefer,vgitmx,vitabs,vitesa(3),vitrel,vitrad,
     +                  vitszn,vittot,vphase,xaltfn,xazmfn,
     +                  xcharg,xflutr,xincli,xlatfn,xlatit,xlonfn,
     +                  xmasse,xorbit(13),xpenfn,xvitfn,zapoge,zperig,
     +                  zromod,tnavig,tguida,tpilot,tpredi,tinteg,
     +                  xlongi,g0terr,pnorme,coefrp,vitson
c
      common / capsul / srefer,vgitmx,xmasse
      common / carext / altcst,dzalim,gaindh
      common / estiro / lambda
      common / gravit / g0terr
      common / intgui / intrgu(2)
      common / intnav / intrnv(2)
      common / orbvis / zapoge,zperig,demiax,excorb,xincli,gomega
      common / period / tnavig,tguida,tpilot,tpredi,tinteg
      common / pernav / dispos(3),disvit(3),disdra
      common / phagui / vphase
      common / missio / xaltfn,xlonfn,xlatfn,xvitfn,xpenfn,xazmfn
      common / modatm / cstgam,facech,rozmod,rmoyen,zromod
c
      common / corgui / dvabrl
      common / intatm / kintat
c
      intrinsic  dsin,dsqrt
c
      external  pnorme
c
      indext = 0
      icrash = 0
c
c		addition des erreurs de navigation (biais constants)
c
      do  i = 1,3
          positn(i) = positr(i) + dispos(i)
          vitesn(i) = vitesr(i) + disvit(i)
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
c		determination de la trainee mesuree acdram
c
      imodel = 0
      call  conphy (positr,vitesr,alfcom,temsim,imodel,
     +              incrar,incrat,
     +              coefan,xcharg,xflutr,pdynan,acdrag,aclift)

c
      acdram = acdrag + disdra
c
c		determination des coefs aeros estimes
c
      imodel = 1
      call  conphy (positn,vitesn,alfcom,temsim,imodel,
     +              incrar,incrat,
     +              coefan,xcharg,xflutr,pdynan,acdrag,aclift)
c
      roesti = 2.d0*dabs(acdram)*xmasse/(coefan(1)*srefer*vitesn(1)**2)

c
c		densite atmospherique (modele embarque guidage)
c  
      call  frayon (positn,
     +              altitu,xlatit)
c
      call  fatmos (altitu,xlatit,xlongi,temsim,imodel,
     +              kintat,
     +              rorefr,vitson)

c
c il y avait un probleme pour coefan(1) a verifier... c'etait fatmos
c 
        call  conphy (positn,vitesn,alfcom,temsim,imodel,
     +                incrar,incrat,
     +                coefan,xcharg,xflutr,pdynan,acdrag,aclift)
        
c
      coefro = (1.d0 - lambda)*coefro + lambda*(roesti/rorefr)
    
      if (altitu.gt.100.d3)then
         coefro = 1.d0
      endif
      
      coefrp = coefro
      roguid = coefro*rorefr
c
c
c		accelerations de trainee et de portance estimees
c
      acdrag = roguid*srefer*(vitesn(1)**2)*coefan(1)/(2.d0*xmasse)
      aclift = roguid*srefer*(vitesn(1)**2)*coefan(2)/(2.d0*xmasse)
            
      pdynan = 0.5*roguid*(vitesn(1)**2)

      acceln(1) = acdrag
      acceln(2) = aclift
      intrnv(1) = incrar
      intrnv(2) = incrat
      intrgu(1) = intrnv(1)
      intrgu(2) = intrnv(2) 
c
c		determination de la densite atmospherique a la sortie
c
      kintaa = kintat
   
      call  fatmos (altcst,xlatit,xlongi,temsim,imodel,
     +              kintaa,
     +              rorfex,vitson)
      roexit = coefro*rorfex
c
c		parametres energetiques estimes
c
      call  energi (positn,vitesn,
     +              energn,vitszn,vittot)
c
c		ecart courant sur les contraintes finales a e i W
c
      call  orbito (positn,vitesn,
     +              xorbit)
c
      ecartn(1) = xorbit(1) - demiax
      ecartn(2) = xorbit(2) - excorb
      ecartn(3) = xorbit(3) - xincli
      ecartn(4) = xorbit(4) - gomega
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
 1000 format(1x,'Passage en phase de sortie  T = ',f8.3,' s')
c

       
      return
      end
