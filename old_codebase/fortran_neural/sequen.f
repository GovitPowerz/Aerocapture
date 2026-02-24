c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : sequen.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module gere les appels aux differents modules du simulateur, a
c3    savoir navigation, guidage, pilotage pour ce qui concerne les algo
c3    rithmes du programme de vol.
c3    L'integration de la trajectoire reelle ainsi que la sauvegarde des
c3    resultats intermediaires se font a la meme cadence.
c3
c3......................................................................
c5    variables d'entree-sortie
c5
c5    temsim            R8   temps courant sur la simulation
c5    datnav            R8   temps courant sur une periode de navigation
c5    datgui            R8   temps courant sur une periode de guidage
c5    datpil            R8   temps courant sur une periode de pilotage
c5    datpho            R8   temps courant sur une periode d'instantannes
c5    icalln            I4   indicateur d'appel a la navigation
c5    icallg            I4   indicateur d'appel au guidage
c5    icallp            I4   indicateur d'appel au pilotage
c5    icallr            I4   indicateur d'integration de trajectoire
c5    icalls            I4   indicateur de sauvegarde resultats courants
c5    icallf            I4   indicateur de prise de photo trajectoire
c5    idebut            I4   indicateur d'initialisation sequentiel
c5......................................................................
c8    composants appelants
c8
c8    simmsr            INT  simulation de l'aerocapture
c8......................................................................
c10   commons utilises
c10
c10   modres                 indicateur de sauvegarde des resultats
c10   oritem                 origine des temps
c10   period                 cadences (GNC, integration)
c10   tkodak                 cadence photo trajectoire
c10   vlimit                 seuil de comparaison
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  sequen (temsim,datnav,datgui,datpil,datpho,
     +                    icalln,icallg,icallp,icallr,icalls,icallf,
     +                    idebut,indrvr,trevrs,temrol,iguida)
c
      implicit none
c
      integer  icalln,icallg,icallp,icallr,icalls,icallf,idebut,
     +         isauve,indrvr,iguida(2),natpil
c
      double precision  temsim,datnav,datgui,datpil,datpho,
     +                  datini,epsiln,tnavig,tguida,tphoto,tpilot,
     +                  tpredi,tinteg,trevrs,temrol
c
      common / modpil / natpil
      common / modres / isauve
      common / oritem / datini
      common / period / tnavig,tguida,tpilot,tpredi,tinteg
      common / tkodak / tphoto
      common / vlimit / epsiln
c
      intrinsic dabs
c
      if (idebut.eq.1) then
c
c		initialisation du sequentiel
c
         temsim = datini
         datnav = 0.d0
         datgui = 0.d0
         datpil = 0.d0
         datpho = 0.d0
         temrol = 0.d0
c
         idebut = 0
         icalln = 1
         icallg = 1
         icallp = 1
         icallf = 1
         
         iguida(1) = 1
         iguida(2) = 1
c
      else
c
c		incrementation des temps courants
c
         temsim = temsim + tinteg
         datgui = datgui + tinteg
         datpil = datpil + tinteg
         datnav = datnav + tinteg
         datpho = datpho + tinteg
c
c		desactivation du guidage longi en roll-reversal
c
         if (indrvr.eq.1) then
            temrol = temrol + tinteg
            if ((trevrs - temrol).le.epsiln) then
               indrvr    = 0
               iguida(1) = 1
               iguida(2) = 1
               temrol    = 0.d0
            else
               iguida(1) = 0
               iguida(2) = 0
            endif         
         endif
c
c		appel de la navigation
c
         if (dabs(datnav - tnavig).le.epsiln) then
            datnav = 0.d0
            icalln = 1
         else
            icalln = 0
         endif
c
c		appel du guidage
c
         if (dabs(datgui - tguida).le.epsiln) then
            datgui = 0.d0
            icallg = 1
         else
            icallg = 0
         endif
c
c		appel du pilotage
c
         if (natpil.eq.0) then
            if (dabs(datpil - tpilot).le.epsiln) then
               datpil = 0.d0
               icallp = 1
            else
               icallp = 0
            endif
         else
            datpil = datgui
            icallp = 1
         endif
c
c
c
         if (dabs(datpho - tphoto).le.epsiln) then
            datpho = 0.d0
            icallf = 1
         else
            icallf = 0
         endif
c
      endif
c
      icallr = 1
      icalls = isauve
c
      return
      end
