c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : .f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine l'acceleration de pesanteur locale radiale et
c3    transverse en fonction de la latitude courante (hypothese de terre
c3    spherique avec J2).
c3
c3......................................................................
c4    variables d'entree
c4
c4    rayvec            R8    rayon vecteur courant
c4    xlatit            R8    latitude courante
c4......................................................................
c6    variables de sortie
c6
c6    gravtl            R8    acceleration de pesanteur laterale
c6    gravtr            R8    acceleration de pesanteur radiale
c6......................................................................
c8    composants appelants
c8
c8    realit            INT   integration trajectoire reelle
c8    trajec            INT   prediction de trajectoire
c8......................................................................
c10   commons utilises
c10
c10   geoide                  caracteristiques champ de pesanteur
c10   planet                  caracteristiques planete cible
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  fgravi (rayvec,xlatit,
     +                    gravtl,gravtr)
c
      implicit none
c
      double precision  rayvec,xlatit,gravtl,gravtr,
     +                  excent,requat,rpolar,xj2,xmug,xomega
c
      common / geoide / excent,xj2,xmug
      common / planet / xomega(3),requat,rpolar
c
      intrinsic  dcos,dsin
c
c		composante laterale
c
      gravtl = 3.d0*xmug*xj2*requat**2*dsin(xlatit)*dcos(xlatit)/
     +         rayvec**4
c
c		composante radiale
c
      gravtr = xmug/rayvec**2 +
     +         3.d0*xmug*xj2*requat**2*(1. d0 - 3.d0*dsin(xlatit)**2)/
     +        (2.d0*rayvec**4)
c
      return
      end
