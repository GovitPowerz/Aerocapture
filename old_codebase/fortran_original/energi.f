c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : energi.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine les parametres energetiques courants sur la
c3    trajectoire
c3
c3......................................................................
c4    variables d'entree
c4
c4    xposit(3)         R8    position absolue repere geocentrique
c4    xvites(3)         R8    vitesse relative repere local
c4......................................................................
c6    variables de sortie
c6
c6    xenerj            R8    energie totale
c6    vitrad            R8    vitesse radiale
c6    vittot            R8    vitesse totale
c6......................................................................
c7    variables internes
c7
c7    altitu            R8    altitude
c7    penvit            R8    pente vitesse
c7    vitrel            R8    vitesse relative
c7    xlatit            R8    latitude
c7......................................................................
c8    composants appelants
c8
c8    naviga            INT   algorithme de navigation
c8    realit            INT   integration trajectoire reelle
c8......................................................................
c9    composants appeles
c9
c9    frayon            INT   caracteristiques geoide
c9    enrtot            INT   energie cinetique et potentielle
c9......................................................................
c10   commons utilises
c10
c10   gravit                  accelerations de pesanteur
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  energi (xposit,xvites,
     +                    xenerj,vitrad,vittot)
c
      implicit none
c
      double precision  xposit(3),xvites(3),xenerj,vitrad,vittot,
     +                  altitu,g0terr,g0mars,penvit,vitrel,xlatit,
     +                  enrtot
c
      common / gravit / g0terr,g0mars
c
      intrinsic  dsin,dsqrt

      external  enrtot
c
c		calculs preliminaires
c
      vitrel = xvites(1)
      penvit = xvites(2)
c
      call  frayon (xposit,
     +              altitu,xlatit)
c
c		energie totale
c
      xenerj = enrtot (xposit,xvites)
c
c		vitesse radiale
c
      vitrad = vitrel*dsin(penvit)
c
C le calcul de de vittot ne sert à rien et dépend de la planète : on n'a pas le g0 de Jupiter
c		vitesse totale
c      vittot = dsqrt(vitrel**2 + 2.d0*g0mars*altitu)
      vittot = 0.
c
      return
      end
