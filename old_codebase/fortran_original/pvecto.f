c1
c1    copyright (c) AEROSPATIALE 1993
c1......................................................................
c2    nom    : pvecto.f
c2    date   : 10/08/93
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module calcule le produit vectoriel
c3......................................................................
c4    variables d'entree
c4
c4    a(3)              R8    vecteur
c4    b(3)              R8    vecteur
c4......................................................................
c6    variables de sortie
c6
c6    c(3)              R8    produit vectoriel de a par b
c6......................................................................
c8    composants appelants
c8
c8    cougyr          INT     calcul des couples gyroscopiques
c8    effort          INT     elaboration du torseur commande
c8    mesure          INT     elaboration des mesures
c8    tirlar          INT     tirage des dispersions aux largages
c8......................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  pvecto (a,b,
     +                    c)
c
c
      implicit none
      double precision   a(3),b(3),c(3)
c
      c(1) = a(2)*b(3) - a(3)*b(2)
      c(2) = a(3)*b(1) - a(1)*b(3)
      c(3) = a(1)*b(2) - a(2)*b(1)
c
      return
      end
