c1
c1    copyright (c) AEROSPATIALE 1993
c1......................................................................
c2    nom    : matvec.f
c2    date   : 10/08/93
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module calcule le produit d'une matrice de dimension (m,n) par
c3    un vecteur de dimension n.
c3
c3......................................................................
c4    variables d'entree
c4
c4    a(m,n)            R8    matrice de dimension m.n
c4    b(n)              R8    vecteur de dimension n
c4    n                 I4    dimension de b
c4    m                 I4    dimension de c
c4......................................................................
c6    variables de sortie
c6
c6    c(m)              R8    vecteur resultat du produit de a par b
c6......................................................................
c8    composants appelants
c8
c8    xvabsl            INT   calcul position-vitesse absolue
c8......................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  matvec (a,b,n,m,
     +                    c)
c
c
      implicit none
      integer  i,j,n,m
c
      double precision  a(m,n),b(n),c(m)
c
      do  i = 1,m
          c(i) = 0.d0
      end do
c
      do  j = 1,n
          do  i = 1,m
              c(i) = c(i) + a(i,j)*b(j)
          end do
      end do
c
      return
      end
