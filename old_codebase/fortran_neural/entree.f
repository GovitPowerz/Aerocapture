c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : entree.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module permet de prendre en compte les choix utilisateur concer
c3    nant le mode d'utilsiation de l'outil de simulation ainsi que les
c3    noms des fichiers de donnees (sous le repertoire ../donnees/) et
c3    de resultats (sous le repertoire ../sorties/).
c3
c3    NOTA  A ajouter: test sur l'existance des fichiers de donnees au
c3          lancement de la simulation (instruction inquire)
c3
c3......................................................................
c6    variables de sortie
c6
c6    xgalea            R8   generateur aleatoire entre 0 et 1
c6    icarlo            I4   indicateur de fonctionnement en Monte-Carlo
c6    itirag            I4   indicateur de lecture ou tirage des disper-
c6                           sions
c6    nbsimu            I4   nombre de simulations a jouer
c6......................................................................
c7    variables internes
c7
c7    iconfr            I4   confirmation des parametres de simulation
c7......................................................................
c8    composants appelants
c8
c8    cisimu            INT  conditions generales de simulation
c8......................................................................
c9    composants appeles
c9
c9    strlen            INT  longueur d'une chaine de caracteres
c9......................................................................
c10   commons utilises
c10
c10   fensim                 numeros de simulations a visualiser
c10   numsim            I4   numero de simulation a rejouer
c10   numvis            I4   numero de simulation a visualiser
c10
c10   modecr
c10   iecran            I4   indicateur d'edition ecran resultats inter
c10                          mediaires
c10
c10   modres
c10   isauve            I4   indicateur de sauvegarde des resultats
c10
c10
c10   ficdat                 suffixes des noms de fichiers de donnees
c10   ficres                 suffixes des noms de fichiers de resultats
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  entree  (xgalea,icarlo,itirag,nbsimu)
c
      implicit none

      integer  icarlo,iconfr,iecran,isauve,istats,itirag,nbsimu,numsim,
     +         numvis,natsim,natpla,irefer,natman,
     +         strlen,natgnn

      double precision  xgalea,gitref,xomega,requat,rpolar,xmulti,
     +                  rhomes(10),altmes(10),giterf

      character  *72 sufaer,sufatm,sufdis,sufgui,sufinc,suflot,sufmis,
     +               sufmsr,sufnav,sufren,sufres,sufsuc,sufgnn

      common / fensim / numsim,numvis
      common / modecr / iecran
      common / modgui / natsim,natgnn
      common / modaga / natman
      common / modres / isauve
      common / muldis / xmulti(4)
      common / ficdat / sufaer,sufatm,sufdis,sufgui,sufinc,suflot,
     +                  sufmis,sufmsr,sufnav,sufren,sufsuc,sufgnn
      common / ficres / sufres
      
      common / traref / irefer
      common / gitrfr / gitref
      
      common / planet / xomega(3),requat,rpolar,natpla
      
      common / rafrai / rhomes,altmes,giterf
  
      external  strlen
c
      xomega(1) = 0.
      xomega(2) = 0.
      xomega(3) = 0.
      requat    = 0.
      rpolar    = 0.
c
c
c		choix des parametres de simulation
c
      write(6,*)
      write(6,1000)
      write(6,1001)
      write(6,1002)
      write(6,1001)
      write(6,1000)
      write(6,*)
      write(6,*)

      iconfr = 0

      do while (iconfr.eq.0)
         write(6,4067)
         read(5,*)      natman
         write(6,4052)
         read (5,*)     natpla 
         write(6,3000)
         read (5,*)     nbsimu
         write(6,3010)
         read (5,*)     natsim
         write(6,3008)
         read (5,*)     natgnn
         write(6,3008)
         read (5,*)     istats
         write(6,3001)
         read (5,*)     itirag
         write(6,3002)
         read (5,*)     numsim
         write(6,3009)
         read (5,*)     isauve
         write(6,3003)
         read (5,*)     numvis
         write(6,3007)
         read (5,*)     iecran
         write(6,3004)
         read (5,*)     xgalea
         write(6,4050)
         read (5,*)     irefer
         write(6,4051)
         read (5,*)     gitref
         giterf = gitref
         write(6,4053)
         read (5,*)     xmulti(1)
         write(6,4054) 
         read (5,*)     xmulti(2)
         write(6,4055)
         read (5,*)     xmulti(3)
         write(6,4056) 
         read (5,*)     xmulti(4)
         write(6,4000)
         read (5,*)     sufmsr
         write(6,4001)
         read (5,*)     sufren
         write(6,4002)
         read (5,*)     sufmis
         write(6,4009)
         read (5,*)     sufgui
         write(6,4009)
         read (5,*)     sufgnn
         write(6,4010)
         read (5,*)     sufinc
         write(6,4004)
         read (5,*)     sufaer
         write(6,4005)
         read (5,*)     sufatm
         write(6,4006)
         read (5,*)     sufdis
         write(6,4007)
         read (5,*)     sufnav
         write(6,4008)
         read (5,*)     suflot
         write(6,4011)
         read (5,*)     sufsuc
         write(6,4020)
         read (5,*)     sufres
c
c		conformite des parametres generaux de simulation
c
         if (nbsimu.ge.2) then
            icarlo = 1
         else
            icarlo = 0
         endif
         if (numsim.le.0) then
            numsim = 1
         endif
         if (numvis.le.0) then
            numvis = 1
         endif
         if (nbsimu.eq.1) then
            numvis = 1
         endif

         write(6,*)
         write(6,6000) nbsimu
         write(6,*)
         write(6,6001) sufmsr(1:strlen(sufmsr))
         write(6,6002) sufren(1:strlen(sufren))
         write(6,6003) sufmis(1:strlen(sufmis))
         write(6,6011) sufgui(1:strlen(sufgui))
         write(6,6012) sufgui(1:strlen(sufinc))
         write(6,6005) sufaer(1:strlen(sufaer))
         write(6,6006) sufatm(1:strlen(sufatm))
         write(6,6007) sufdis(1:strlen(sufdis))
         write(6,6008) sufnav(1:strlen(sufnav))
         write(6,6009) suflot(1:strlen(suflot))
         write(6,6013) sufsuc(1:strlen(sufsuc))
         write(6,6010) sufres(1:strlen(sufres))
         write(6,*)

         write(6,5000)
         read (5,*)     iconfr
         if ((icarlo.eq.1).and.(natsim.eq.4)) then
            write(6,*) 'cas de simulation non envisage'
            iconfr = 0
         endif
         if ((natpla.le.2).and.(natpla.ge.6)) then
            iconfr = 0
         endif
c
      end do
c
      if (istats.eq.1) then
         icarlo = 2
         itirag = 0
      endif
c
c		formats de lecture-ecriture
c
 1000 format(10x,'################################')
 1001 format(10x,'#                              #')
 1002 format(10x,'#   AEROCAPTURE -  CAPREE      #')
 1003 format(10x,'#         AGA -  CAPREE        #')
 1004 format(10x,'#   AEROCAPTURE -  Jupiter     #')
c
 3000 format(1x,'nombre de simulations a jouer')
 3008 format(1x,'traitement statistique uniquement (1)')
 3001 format(1x,'lecture (0) ou creation (1) des dispersions')
 3002 format(1x,'numero de simulation a rejouer')
 3003 format(1x,'numero de simulation a visualiser')
 3004 format(1x,'generateur aleatoire entre 0 et 1')
 3007 format(1x,'edition ecran messages intermediaires (1) ou non (0)')
 3009 format(1x,'sauvegarde des resultats (1) ou non (0)')
 3010 format(1x,'totalite du guidage (1), capture (2) ou sortie (3)')

 4000 format(1x,'suffixe fichier caracteristiques capsule')
 4001 format(1x,'suffixe fichier conditions a la rentree')
 4002 format(1x,'suffixe fichier caracteristique mission aerocapture')
 4004 format(1x,'suffixe fichier tables aerodynamiques')
 4005 format(1x,'suffixe fichier tables atmospheriques')
 4006 format(1x,'suffixe fichier dispersions initiales')
 4007 format(1x,'suffixe fichier performances navigation')
 4008 format(1x,'suffixe fichier tirage dispersions')
 4009 format(1x,'suffixe fichier caracteristiques guidage')
 4010 format(1x,'suffixe fichier profil d''incidence commandee')
 4011 format(1x,'suffixe fichier erreurs finales admissibles')
 4020 format(1x,'suffixe fichiers de resultats',a30)
 4050 format(1x,'trajectoire de reference (1) ou guidee (0)')
 4051 format(1x,'gite constante sur la reference (deg)')
 4052 format(1x,'nature Planete: Terre 3  Mars 4  Jupiter  5')
 4053 format(1x,'coeff. multi des dispersions de nav aerocapture')
 4054 format(1x,'coeff. multi des dispersions de nav interplanetaire')
 4055 format(1x,'coeff. multi des dispersions mesure accelero')
 4056 format(1x,'coeff. multi des dispersions aerodynamiques')
 4067 format(1x,'aerocapture (1) ou aero-gravity-assist (2)')
 
c
 6000 format(1x,'nombre de simulations   ',i4)
 6001 format(1x,'fichier caracteristiques capsule    : ',a30)
 6002 format(1x,'fichier conditions a la rentree     : ',a30)
 6003 format(1x,'fichier caracteristiques mission    : ',a30)
 6005 format(1x,'fichier tables aerodynamiques       : ',a30)
 6006 format(1x,'fichier tables atmospheriques       : ',a30)
 6007 format(1x,'fichier dispersions initiales       : ',a30)
 6008 format(1x,'fichier performances navigation     : ',a30)
 6009 format(1x,'fichier tirage dispersions          : ',a30)
 6011 format(1x,'fichier caracteristiques guidage    : ',a30)
 6012 format(1x,'fichier profil incidence commandee  : ',a30)
 6013 format(1x,'fichier erreurs finales admissibles : ',a30)
 6010 format(1x,'fichier resultats                   : ',a30)
c
 5000 format(1x,'Confirmation des parametres de simulation (0 - 1)?')
c
      return
      end
