c1
c1    copyright (c) EADS Launch Vehicles 2002
c1......................................................................
c2    nom    : .f
c2    date   : 01/07/02
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine
c3
c3......................................................................
c4    variables d'entree
c4
c4    varia             I4    nom
c4......................................................................
c5    variables d'entree-sortie
c5
c5......................................................................
c6    variables de sortie
c6
c6......................................................................
c7    variables internes
c7
c7......................................................................
c8    composants appelants
c8
c8......................................................................
c9    composants appeles
c9
c9......................................................................
c10   commons utilises
c10
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  tbgain (altitu,coefan,alfcom,
     +                    gaindh,gainpd,coefpd)
c
      implicit none
c
      integer  i,inumer,nzapd
      
      integer  nzapdmx
      parameter (nzapdmx = 1000)
c
      double precision  altitu,coefan(2),alfcom,gaindh,gainpd,
     +                  coefpd(2),
     +                  amorft,pulsft,altpdn,tabpda,tabpdb,srefer,
     +                  vgitmx,xmasse
c
      common / capsul / srefer,vgitmx,xmasse
      common / gainmu / amorft,pulsft
      common / modpdn / nzapd
      common / varpdn / altpdn(nzapdmx),tabpda(nzapdmx),tabpdb(nzapdmx)
c
c		recherche de la tranche d'altitude consideree
c
      inumer = 0
      do  i = 1,nzapd-1
          if (((altitu/1.d3).ge.altpdn(i)).and.
     +        ((altitu/1.d3).lt.altpdn(i+1)).and.
     +        (inumer.eq.0)) then
             inumer = i
          endif
      end do
      if (inumer.eq.0) then
         inumer = nzapd
      endif
c
c		modele de pression dynamique
c
      coefpd(1) = tabpda(inumer)
      coefpd(2) = tabpdb(inumer)
c
c		gains de l'asservissement
c
      gaindh =-2.d0*amorft*pulsft*xmasse/(srefer*coefan(2)) 
      gainpd =-pulsft**2*xmasse/(coefpd(1)*srefer*coefan(2))
c
      return
      end
