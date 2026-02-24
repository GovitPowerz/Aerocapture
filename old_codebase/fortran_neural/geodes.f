c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : geodes.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module permet de passer des coordonnees geodesiques aux coordon
c3    nees geocentrique, dans un systeme de representation spherique.
c3
c3......................................................................
c4    variables d'entree
c4
c4    altitu            R8    altitude geodesique
c4    xlatit            R8    latitude geodesique 
c4    xlongi            R8    longitude geodesique 
c4......................................................................
c6    variables de sortie
c6
c6    xposit(3)         R8    position geocentrique spherique
c6......................................................................
c8    composants appelants
c8
c8......................................................................
c10   commons utilises
c10
c10   planet                  caracteristiques planete
c10   vlimit                  seuil de comparaison
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  geodes (altitu,xlatit,xlongi,
     +                    xposit)
c
      implicit none
c
      double precision  altitu,xlatit,xlongi,xposit(3),
     +                  coslat,coslon,epsiln,excent,positx(3),requat,
     +                  rpolar,sinlat,sinlon,xj2,xmug,xomega,xrayon,
     +                  pnorme 
c
      common / geoide / excent,xj2,xmug
      common / planet / xomega(3),requat,rpolar
      common / vlimit / epsiln
c
      external  pnorme
c
      intrinsic  dabs,datan2,dcos,dsin,dsqrt 
c
      coslat = dcos(xlatit)
      sinlat = dsin(xlatit)
      coslon = dcos(xlongi)
      sinlon = dsin(xlongi)
c
      xrayon = requat/dsqrt(1.d0 - excent**2*sinlat**2)
c
c		coordonnees cartesiennes geocentriques
c
      positx(1) =(xrayon + altitu)*coslat*coslon      
      positx(2) =(xrayon + altitu)*coslat*sinlon      
      positx(3) =(xrayon*(1.d0 - excent**2) + altitu)*sinlat      
c      
c		coordonnees spheriques geocentriques
c
      xposit(1) = pnorme (positx)
      xposit(2) = xlongi
c
      if (dabs(coslon).le.epsiln) then
         xposit(3) = datan2(positx(3)/xposit(1),
     +                      positx(2)/(xposit(1)*sinlon))
      else
         xposit(3) = datan2(positx(3)/xposit(1),
     +                      positx(1)/(xposit(1)*coslon))      
      endif
c
      return
      end 
