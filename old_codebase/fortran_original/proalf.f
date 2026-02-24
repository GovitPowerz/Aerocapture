c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : .f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Cette fonction determine la valeur courante en X du profil d'inci-
c3    dence commandee
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
c6    proalf            R8    valeur courante en X du profil
c6......................................................................
c8    composants appelants
c8
c8    guialf            INT   consigne de guidage  en incidence
c8......................................................................
c9    composants appeles
c9
c9......................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      function  proalf (positn,vitesn,roguid)
c
      implicit none
c
      double precision  positn(3),vitesn(3),roguid,proalf,
     +                  altitu,pdynam,xlatit
c
      pdynam = roguid*vitesn(1)**2
c
      call  frayon (positn,
     +              altitu,xlatit)
c
      proalf = altitu
c
      return
      end
