c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : pnorme.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Cette fonction determine la norme d'un vecteur de dimension 3
c3
c3......................................................................
c4    variables d'entree
c4
c4    vector(3)         R8    vecteur
c4......................................................................
c6    variables de sortie
c6
c6    pnorme            R8    norme du vecteur
c6......................................................................
c8    composants appelants
c8
c8    orbito            INT   calcul des parametres orbitaux
c8......................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      function  pnorme (vector)
c
      implicit none
c
      double precision  pnorme,vector(3)
c
      intrinsic  dsqrt
c
      pnorme = dsqrt(vector(1)**2 + vector(2)**2 + vector(3)**2)
c
      return
      end
