c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : cartes.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module permet d'exprimer un vecteur fourni initialement en coor
c3    donnees spheriques en coordonnees cartesiennes.
c3
c3    NOTA  Dans le cas d' un vecteur position, les composantes du vec-
c3          teur sont: altitude geocentrique, longitude, latitude.
c3          Danns le cas d'un vecteur vitesse, les composantes du vec-
c3          teur sont: norme de la vitesse, pente, azimut.
c3
c3......................................................................
c4    variables d'entree
c4
c4    xspher(3)         R8    vecteur en coordonnees spheriques (norme,
c4                            pente, azimut ou longitude,latitude)
c4    iposvi            I4    indicateur de vecteur position (0) ou vi-
c4                            tesse (1)
c4......................................................................
c6    variables de sortie
c6
c6    xcarte(3)         R8    vecteur en coordonnees cartesiennes
c6......................................................................
c7    variables internes
c7
c7    anglxy            R8    angle dans le plan xy (azimut, longitude)
c7    anglxz            R8    angle dans le plan xz (pente, latitude)
c7......................................................................
c8    composants appelants
c8
c8    xvabsl            INT   position-vitesse absolues
c8......................................................................
c10   commons utilises
c10
c10   trigon                  constantes trigonometriques
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  cartes (xspher,iposvi,
     +                    xcarte)
c
      implicit none
c
      integer  iposvi
c
      double precision  xspher(3),xcarte(3),
     +                  anglxy,anglxz,degrad,pi
c
      common / trigon / degrad,pi
c
      intrinsic  dcos,dsin
c
c		choix des angles
c
      if (iposvi.eq.1) then
         anglxy =-xspher(3) + 2.d0*pi
         anglxz = xspher(2)
      else
         anglxy = xspher(2)
         anglxz = xspher(3)
      endif
c
c		passage en coordonnees cartesiennes
c
      xcarte(1) = xspher(1)*dcos(anglxz)*dcos(anglxy)
      xcarte(2) = xspher(1)*dcos(anglxz)*dsin(anglxy)
      xcarte(3) = xspher(1)*dsin(anglxz)
c
      return
      end
