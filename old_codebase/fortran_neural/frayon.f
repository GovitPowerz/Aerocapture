c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : frayon.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine l'altitude et la latitude geodesique a partir
c3    de la position geocentrique. 
c3
c3......................................................................
c4    variables d'entree
c4 
c4    positn(3)         R8    position absolue geocentrique spherique
c4......................................................................
c6    variables de sortie
c6
c6    altitu            R8    altitude geodesique
c6    xlatit            R8    latitude geodesique
c6......................................................................
c7    variables internes
c7
c7......................................................................
c8    composants appelants
c8
c8......................................................................
c10   commons utilises
c10
c10   planet                  caracteristiques planete
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  frayon (positn,
     +                    altitu,xlatit)
c
      implicit none
c
      integer  iterat,
     +         ialtit
c
      double precision  positn(3),altitu,xlatit,
     +                  altitr,altitz,coslat,excent,positp,positr,
     +                  positx,posity,positz,requat,rplant,rpolar,
     +                  sinlat,tanlat,vlimit,xj2,xmug,xomega
c
      common / geoide / excent,xj2,xmug
      common / planet / xomega(3),requat,rpolar
c
      intrinsic  dabs,datan2,dcos,dsign,dsin,dsqrt
c
      coslat = dcos(positn(3))
      sinlat = dsin(positn(3))
      vlimit = 1.d-2
c
c		passage en coordonnees cartesiennes
c
      positx = positn(1)*dcos(positn(3))*dcos(positn(2))
      posity = positn(1)*dcos(positn(3))*dsin(positn(2))
      positz = positn(1)*dsin(positn(3))  
c
      positp  = dsqrt(positx**2 + posity**2)
      positr  = dsqrt(positz**2 + positp**2)
c
      altitr = positr - requat
c
      if (requat.eq.rpolar)then
c
c		hypothese de planete spherique
c
         altitu = altitr
         sinlat = positz/positr
         coslat = positp/positr
         xlatit = datan2(sinlat,
     +                   coslat)
c
      else
c
c		hypothese de planete non spherique
c
         rplant = requat
         altitz = altitr - dsqrt(requat*rpolar)
         iterat = 0
c
c		determination iterative de l'altitude
c
         ialtit = 0
c
         do  while (ialtit.eq.0)
c
             iterat = iterat + 1
c
             tanlat = (positz/positp)/
     +                (1.d0 - excent**2*rplant/(rplant + altitz))
             sinlat = dsqrt(tanlat**2/
     +                     (1.d0 + tanlat**2))
             coslat = dsqrt(1.d0/
     +                     (1.d0 + tanlat**2))
             altitu = positp/coslat - rplant
             sinlat = dsign(sinlat,tanlat)
c
             if ((dabs(altitu - altitz).lt.vlimit).or.
     +           (iterat.ge.10)) then
                xlatit = datan2(sinlat,
     +                         coslat)
                ialtit = 1
             else
                rplant = requat/
     +                   dsqrt(1.d0 -excent**2*sinlat**2)
                altitz = altitu
                ialtit = 0
             endif
         end do
      endif
c
      return
      end 
