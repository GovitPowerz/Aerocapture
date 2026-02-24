c1
c1    copyright (c) AEROSPATIALE 1999
c1......................................................................
c2    nom    : inimsr.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module realise l'initialisation des parametres utilises par le
c3    simulateur (mises a zero, calculs preliminaires, addiiton de dis-
c3    persions,...)
c3
c3    NOTA  on initialise la gite de la capsule a 0.
c3......................................................................
c4    variables d'entree
c4
c4    icarlo            I4    indicateur de mode Monte-Carlo
c4    isimul            I4    numero de simulation
c4......................................................................
c5    variables d'entree-sortie
c5
c5    xorbit(7)         R8    parametres orbitaux
c5    positr(3)         R8    position reelle repere geocentrique
c5    vitesr(3)         R8    vitesse relative reelle repere local
c5    positn(3)         R8    position estimee repere geocentrique
c5    vitesn(3)         R8    vitesse relative estimee repere local
c5    altmax(3)         R8    altitude de flux, facteur de charge et Pd
c5                            max
c5    datmax(3)         R8    instants de flux, facteur de charge et Pd
c5                            max
c5    fluter(2)         R8    flux thermique courant et max.
c5    fcharg(2)         R8    facteur de charge courant et max
c5    pdynam(2)         R8    pression dynamique courante et max
c5    coefro            R8    coefficient d'estimation de ro
c5    gitpre            R8    gite precedente
c5    sgngit            R8    sigen de la gite commandee
c5    somgit            R8    cumul des increments de commande de gite
c5    somflu            R8    integrale de flux
c5    temsim            R8    temps courant
c5    trebon            R8    date du rebond atmospherique
c5    zrebon            R8    altitude de rebond atmospherique
c5    ibounc            I4    indicateur de rebond (navigation)
c5    icaptr            I4    indicateur de secruisation de capture
c5    idebut            I4    indicateur d'initialisation du sequentiel
c5    iphase            I4    indicqteur d epqhse du guidage longi
c5    iprepr(2)         I4    compteur de securisations du guidage
c5    irebon            I4    indicateur de rebond
c5    isauve            I4    indicateur de sauvegarde
c5    isecur            I4    indicateur de securisation du guidage
c5    nbroll            I4    nombre de renverses de roulis
c5    indrol            I4    indicateur de renverse de roulis
c5    indext            I4    indicateur de guidage en phase de sortie
c5    isorti            I4    indicateur de sauvegarde en phase de sotie
c5......................................................................
c7    variables internes
c7
c7    dispos(3)         R8    dispersions initiales en position
c7    disvit(3)         R8    dispersions initiales en vitesse
c7......................................................................
c8    composants appelants
c8
c8    inimsr            INT   simulation de l'aerocapture
c8......................................................................
c9    composants appeles
c9
c9    etaini            INT   edition ecran des conditions initiales
c9    orbito            INT   parametres orbitaux
c9......................................................................
c10   commons utilises
c10
c10   fensim                  numeros de simulation a rejouer...
c10   oricom                  angles commandes initiaux
c10   trigon                  constantes trigonometriques
c10   xvrent                  conditions nominales a la rentree
c10
c10   intalf
c10   kintal            I4    increment d'interpolation incidence
c10
c10   intgui
c10   kintgu(2)         I4    increments d'interpolation guidage
c10
c10   intnav
c10   kintgu(2)         I4    increments d'interpolation navigation
c10
c10   intrea
c10   kintgu(2)         I4    increments d'interpolation trajectoire
c10
c10   mecaer
c10   distam            R8    coef. de dispersion densite atmospherique
c10   dxdrag            R8    coef. de dispersion sur la trainee
c10   dxlift            R8    coef. de dispersion sur la portance
c10
c10   perinj
c10   dxposi(3)         R8    erreurs en position a l'injection
c10   dxvite(3)         R8    erreurs en vitesse a l'injection
c10
c10   pernav
c10   dispos(3)         R8    gabarit erreurs de navigation en posiiton
c10   disvit(3)         R8    gabarit erreurs de navigation en vitesse
c10   disacd            R8    gabarit erreur mesure acceleration trainee
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  inimsr (icarlo,isimul,
     +                    xorbit,ecartr,positr,vitesr,positn,vitesn,
     +                    altmax,datmax,fluter,fcharg,pdynam,alfcom,
     +                    coefro,gitpre,sgngit,somflu,somgit,temsim,
     +                    trebon,vitpre,vitref,zrebon,iprepr,ibounc,
     +                    icaptr,idebut,ifinal,iphase,irebon,isauve,
     +                    isecur,nbroll,indrol,indext,isorti,iguida,
     +			  ilater,tlater,dtroll,itera,gpilpr,gitpil)
c
      implicit none
c
      include '../include/dimensions.incl'
c
      integer  icarlo,isimul,ibounc,icaptr,idebut,ifinal,iphase,
     +         iprepr(2),irebon,isauve,isecur,nbroll,itera,
     +         i,kintal,kintgu,kintnv,kinttr,nbalfa,nbmach,
     +         numsim,numvis,indrol,indext,isorti,natsim,
     +         kintop,kintat,kintlp,ilater,iguida(2)
c
      double precision  xorbit(13),ecartr(4),positr(3),vitesr(3),
     +                  positn(3),vitesn(3),altmax(3),datmax(3),
     +                  fluter(2),fcharg(2),pdynam(2),alfcom,coefro,
     +                  gitpre,sgngit,somflu,somgit,temsim,trebon,
     +                  vitpre,xaleat,zrebon,
     +                  alfini,dalfae,daltit,datini,dazimu,dcxeng,
     +                  dczeng,ddensi,degrad,demiax,disacd,dlatit,
     +                  dlongi,dnalti,dnazim,dndrag,dnlati,dnlong,
     +                  dnpent,dnvite,dpente,dxposi,dxvite,disatm,
     +                  dvites,dxdrag,dxlift,dispos,disvit,excorb,
     +                  gitini,gomega,pi,positz,vitref,vitesz,xaltfn,
     +                  xazmfn,xincli,xlatfn,xlonfn,xpenfn,xvitfn,
     +                  zapoge,zperig,gitref,dmvehi,dxmass,
     +			tlater,dtroll,gpilpr,gitpil
c
      common / fensim / numsim,numvis
      common / modalf / nbalfa
      common / modgui / natsim
      common / orbvis / zapoge,zperig,demiax,excorb,xincli,gomega
      common / oritem / datini
      common / tablar / nbmach
      common / trigon / degrad,pi
      common / xvrent / positz(3),vitesz(3)
      common / gitrfr / gitref
c
      common / missio / xaltfn,xlonfn,xlatfn,xvitfn,xpenfn,xazmfn
      common / mecaer / dalfae,disatm,dxdrag,dxlift
      common / oricom / alfini,gitini
      common / perinj / dxposi(3),dxvite(3)
      common / pernav / dispos(3),disvit(3),disacd
      common / mecmas / dxmass
      common / intalf / kintal
      common / intgui / kintgu(2)
      common / intnav / kintnv(2)
      common / intrea / kinttr(2)
      common / intatm / kintat
      common / tabopt / kintop
      common / tabopp / kintlp
      
c
      intrinsic  dsin
c
c		lecture des dispersions
c
      if (icarlo.eq.1) then
         read(108,1000) i,xaleat,
     +                    daltit,dlongi,dlatit,
     +                    dvites,dazimu,dpente,
     +                    ddensi,
     +                    dcxeng,dczeng,
     +                    dnalti,dnlati,dnlong,
     +                    dnvite,dnazim,dnpent,
     +                    dndrag,dalfae,dmvehi
      else
         do  i = 1,numsim-1
             read(108,*)
         end do
         read(108,1000) i,xaleat,
     +                    daltit,dlongi,dlatit,
     +                    dvites,dazimu,dpente,
     +                    ddensi,
     +                    dcxeng,dczeng,
     +                    dnalti,dnlati,dnlong,
     +                    dnvite,dnazim,dnpent,
     +                    dndrag,dalfae,dmvehi
      endif
c
c		erreurs aerodynamiques et atmospheriques
c
      disatm = ddensi
      dxdrag = dcxeng
      dxlift = dczeng
c
c		erreurs massiques
c
      dxmass = dmvehi
c
c		erreurs initiales
c
      dxposi(1) = daltit
      dxposi(2) = dlongi
      dxposi(3) = dlatit
      dxvite(1) = dvites
      dxvite(2) = dpente
      dxvite(3) = dazimu
c
c		erreurs de navigation
c
      dispos(1) = dnalti
      dispos(2) = dnlong
      dispos(3) = dnlati
      disvit(1) = dnvite
      disvit(2) = dnpent
      disvit(3) = dnazim
      disacd    = dndrag
c
c		etat estime (prises en compte erreurs de navigation)
c
      do  i = 1,3
          positn(i) = positz(i) + dispos(i)
          vitesn(i) = vitesz(i) + disvit(i)
      end do
c
c		etat reel (prises en compte erreurs d'injection)
c
      positr(1) = daltit + positz(1)
      positr(2) = dlongi + positz(2)
      positr(3) = dlatit + positz(3)
      vitesr(1) = dvites + vitesz(1)
      vitesr(2) = dpente + vitesz(2)
      vitesr(3) = dazimu + vitesz(3)
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
      gitpre = gitini
      gpilpr = gitpre
      gitpil = gpilpr
      if (gitref.le.0.d0) then
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
      ifinal = 0
      indext =-1
      indrol = 0
      iphase = 1
      irebon = 0
      isecur = 0
      isorti = 0
      nbroll = 0
      itera  = 0
      
      iguida(1) = 1
      iguida(2) = 1
      
      if (isimul.eq.numvis) then
         isauve = 1
      else
         isauve = 0
      endif
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
      if (natsim.eq.3) then
         iphase = 1
         ibounc = 1
      endif
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
      call  etaini (positr,vitesr,isimul)
c
 1000 format(i5,1x,d15.7,18(1x,d15.7))
 1100 format(2(1x,f10.3))
c
      return
      end
