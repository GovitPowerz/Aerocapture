c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : reploc.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module permet de passer du repere local au repere geocentrique
c3    a partir de la position courante du point (exprimee en longitude-
c3    latitude).
c3    Le repere local est defini de la maniere suivante:
c3    - rotation d' angle L (longitude, L > 0 vers l'Ouest) autour de Zt
c3      qui fait passer de (Xt,Yt) a (x1,y1);
c3    - rotation d'angle l (latitude, l > 0 vers le Nord) autour de y1
c3      qui fait passer de (x1,z1=Zt) a (x2,z2);
c3    - permutation des axes pour passer de (x2,y2,z2) a (z0,x0,y0), z0
c3      etant porte par la verticale locale, du bas vers le haut, l'axe\
c3      y0 etant oriente vers le Nord pour l > 0.
c3......................................................................
c4    variables d'entree
c4
c4    posits(3)         R8    position absolue coordonnees spheriques
c4    indloc            I4    indicateur de changement de repere (local
c4                            geocentrique ou geocentrique-local)
c4......................................................................
c6    variables de sortie
c6
c6    plocal(3,3)       R8    matrice de changement de repere
c6......................................................................
c8    composants appelants
c8
c8    orbito            INT   parametres orbitaux
c8......................................................................
c10   commons utilises
c10
c10   vlimit                  seuil de comparaison
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  reploc (posits,indloc,
     +                    plocal)
c
      implicit none
c
      integer  indloc,
     +         i,j
c
      double precision  posits(3),plocal(3,3),
     +                  coslat,coslon,epsiln,sinlat,sinlon
c
      common / vlimit / epsiln
c
      intrinsic  dcos,dsin
c
      sinlat = dsin(posits(3))
      coslat = dcos(posits(3))
      sinlon = dsin(posits(2))
      coslon = dcos(posits(2))
c
      if (indloc.eq.0) then
c
c		changement repere local-repere geocentrique
c
         plocal(1,1) =-coslon*sinlat
         plocal(1,2) = sinlon
         plocal(1,3) = coslon*coslat

         plocal(2,1) =-sinlon*sinlat
         plocal(2,2) =-coslon
         plocal(2,3) = sinlon*coslat

         plocal(3,1) = coslat
         plocal(3,2) = 0.d0
         plocal(3,3) = sinlat
c
      else
c
c		changement repere geocentrique-repere local
c
         plocal(1,1) =-coslon*sinlat
         plocal(1,2) =-sinlon*sinlat
         plocal(1,3) = coslat

         plocal(2,1) = sinlon
         plocal(2,2) =-coslon
         plocal(2,3) = 0.d0

         plocal(3,1) = coslon*coslat
         plocal(3,2) = sinlon*coslat
         plocal(3,3) =        sinlat
c
      endif
c
      do  i = 1,3
          do  j = 1,3
              if (dabs(plocal(i,j)).lt.epsiln**2) then
                 plocal(i,j) = 0.d0
              endif
          end do
      end do
c
      return
      end
