c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : .f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine la consigne en incidence.
c3
c3......................................................................
c4    variables d'entree
c4
c4    positn(3)         R8    position absolue geocentrique estimee
c4    vitesn(3)         R8    vitesse relative locale estimee
c4    roguid            R8    densite atmospherique estimee 
c4......................................................................
c6    variables de sortie
c6
c6    alfcom            R8    incidence commandee
c6......................................................................
c8    composants appelants
c8
c8    guidag            INT   generation des consignes de guidage
c8......................................................................
c9    composants appeles
c9
c9    proalf            INT
c9......................................................................
c10   commons utilises
c10
c10   intalf                  increment pour interpolation
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   non                     parametre de common variable   / intalf /
c11.....................................................................
c
      subroutine  guialf (positn,vitesn,roguid,
     +                    alfcom)
c
      implicit none
c
      include '../include/dimensions.incl'
c
      integer  kintal,nbalfa
c
      double precision  positn(3),vitesn(3),roguid,alfcom,
     +                  paramx,profax,profay,
     +                  proalf
c
      common / intalf / kintal
      common / modalf / nbalfa
      common / loialf / profax(nalfax),profay(nalfax)
c
      external  proalf
c
      paramx = proalf (positn,vitesn,roguid)
c
c		interpolation de l'incidence commandee sur profil
c

      call  intrmo (paramx,profax,profay,nbalfa,
     +              kintal,
     +              alfcom)
c
      return
      end
