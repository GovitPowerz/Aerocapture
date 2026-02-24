c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : pscalr.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Cette fonction determine le produit scalaire de 2 vecteurs
c3
c3......................................................................
c4    variables d'entree
c4
c4    vectra(3)         R8    vecteur
c4    vectrb(3)         R8    vecteur
c4......................................................................
c6    variables de sortie
c6
c6    pscalr            R8    produit scalaire des vecteurs
c6......................................................................
c8    composants appelants
c8
c8......................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      function  pscalr (vectra,vectrb)
c
      implicit none
c
      double precision  vectra(3),vectrb(3),pscalr
c
      pscalr = vectra(1)*vectrb(1) +
     +         vectra(2)*vectrb(2) +
     +         vectra(3)*vectrb(3)
c
      return
      end
