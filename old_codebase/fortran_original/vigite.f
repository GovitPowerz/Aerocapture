c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : vigite.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module realise une saturation de la commande de gite sur un cri
c3    tere de vitesse de gite commandee maximale sur une periode de gui-
c3    dage
c3
c3......................................................................
c5    variables d'entree-sortie
c5
c5    gitcom            R8    gite courante commandee
c5    gitpre            R8    gite commandee precedente
c5    somgit            R8    consommation de gite
c5......................................................................
c6    variables de sortie
c6
c6    vitgit            R8    vitesse de gite
c6    isatur            I4    indiacteur de saturation
c6......................................................................
c8    composants appelants
c8
c8    guidag            INT   guidage par matrice de sensibilite
c8......................................................................
c10   commons utilises
c10
c10   capsul                  caracteristiques capsule
c10   modecr                  edition ecran intermediaires
c10   period                  cadences
c10   vlimit                  seuil de comparaison
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  vigite (gitcom,gitpre,somgit,
     +                    vitgit,isatmu)
c
      implicit none
c
      integer  isatmu,
     +         iecran
c
      double precision  gitcom,gitpre,somgit,vitgit,
     +                  epsiln,srefer,tinteg,tguida,tnavig,tpilot,
     +                  tpredi,vgitmx,xmasse
c
      common / capsul / srefer,vgitmx,xmasse
      common / modecr / iecran
      common / period / tnavig,tguida,tpilot,tpredi,tinteg
      common / vlimit / epsiln
c
      intrinsic  dabs
c
      vitgit =(gitcom - gitpre)/tguida
      isatmu = 0
c
c		test de saturation de la vitesse
c
       if ((dabs(vitgit) - vgitmx).gt.epsiln) then
          isatmu = 1
          if (gitcom.gt.gitpre) then
             gitcom = gitpre + vgitmx*tguida
          else
             gitcom = gitpre - vgitmx*tguida
          endif
       endif
c
c		calcul du cumul de la gite commandee
c
       if (dabs(vitgit).gt.epsiln) then
          somgit = somgit + dabs(gitcom - gitpre)
       endif
c
      gitpre = gitcom
c
      return
      end
