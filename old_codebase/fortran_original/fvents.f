c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : fvents.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine les composantes de la vitesse du vent
c3
c3    NOTA  Actuellement, on ne tient pas compte de vent dans les simula
c3          tions d'aerocapture
c3......................................................................
c4    variables d'entree
c4
c4    altitu            R8    altitude
c4......................................................................
c6    variables de sortie
c6
c6    vventm            R8
c6    vventz            R8
c6......................................................................
c8    composants appelants
c8
c8    realit            INT  integration trajectoir reelle
c8......................................................................
c10   commons utilises
c10
c10   modven                 modelisation du vent
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  fvents (altitu,
     +                    vventm,vventz)
c
      implicit none
c
      integer  ivents
c
      double precision  altitu,vventm,vventz
c
      common / modven / ivents
c
      if (ivents.eq.0) then
         vventm = 0.d0
         vventz = 0.d0
      endif
c
      return
      end
