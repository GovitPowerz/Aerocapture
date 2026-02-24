c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : finmsr.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module permet de decider de l'arret de la simulation d'aerocap-
c3    ture selon 3 criteres, a savoir:
c3    - un critere d'altitude finale (soit celle de sortie d'atmosphere
c3      supposee de 130 km) dans le cas d'un rebond sur les couches den
c3      ses de l'atmosphere lors de la phase de descente;
c3    - un critere d'altitude finale negative dans le cas d'une trajec-
c3      toire avec ou sans rebond,
c3    - un critere de duree de simulation dans le cas d'une trajectoire
c3      d'aerocapture avec rebond sans sortie d'atmosphere (critere ar-
c3      bitraire de 2000 s)
c3
c3    On prevoit de plus le cas ou seule une partie de la phase d'aero-
c3    capture est simulee (phase de capture, vol equilibre, ou phase de
c3    sortie). Seul la premiere configuration donne lieu a un traitement.
c3    Dans ce cas, on arrete la simulation des que l'on change de phase.
c3
c3......................................................................
c4    variables d'entree
c4
c4    altitr            R8    altitude courante
c4    temps             R8    temps courant
c4    vitrad            R8    vitesse radiale
c4    irebon            I4    indicateur de rebond atmospherique
c4......................................................................
c6    variables de sortie
c6
c6    ifinal            I4    indicateur de fin de simulation
c6......................................................................
c8    composants appelants
c8
c8    simmsr            INT  simulation d'aerocapture
c8......................................................................
c10   commons utilises
c10
c10   missio                 caracteristiques mission
c10   modecr                 edition ecran intermediares
c10   modgui                 nature des phases simulees
c10   phagui                 critere de changement de phase du guidage
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  finmsr (altitr,temsim,vitrad,irebon,
     +                    ifinal)
c
      implicit none
c
      integer  irebon,ifinal,
     +         natsim,iecran
c
      double precision  altitr,temsim,vitrad,
     +                  tramax,vphase,xaltfn,xazmfn,xlonfn,xlatfn,
     +                  xpenfn,xvitfn
c
      common / modecr / iecran
      common / modgui / natsim
      common / missio / xaltfn,xlonfn,xlatfn,xvitfn,xpenfn,xazmfn
      common / phagui / vphase
c
      tramax = 5000.
c
c		test sur altitude negative (crash capsule)
c
      if (altitr.le.0.d0) then
         ifinal = 1
         if (iecran.eq.1) write(6,1000) temsim
      endif
c
c		test sur depassement de temps
c
      if (temsim.ge.tramax) then
         ifinal = 2
          if (iecran.eq.1) write(6,2000) altitr/1.d3
      endif
c
c		test sur sortie d'atmosphere
c
      if ((irebon.eq.1).and.(altitr.ge.xaltfn)) then
         ifinal = 3
          if (iecran.eq.1) write(6,3000) temsim
      endif
c
c		cas d'une simulation partielle de l'aerocapture
c
      if (natsim.eq.2) then
         if ((irebon.eq.1).and.(vitrad.ge.vphase)) then
            ifinal = 4
            if (iecran.eq.1)  write(6,4000) temsim
         endif
      endif
c
 1000 format(1x,'Arret sur crash Orbiter a T = ',f8.3,' s')
 2000 format(1x,'Arret sur critere temporel a Z = ',f8.3,' km')
 3000 format(1x,'Arret sur altitude fin mission a T = ',f8.3,' s')
 4000 format(1x,'Arret sur changement de phase a T = ',f8.3,' s')
c
      return
      end
