c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : ergols.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine le cout en DV necessaire pour rejoindre l'orbi
c3    te de parking depuis les conditions de sortie d'aerocapture. La mo
c3    dification de l'orbite atteinte en fin d'aerocapture se fait en 2
c3    manoeuvres propulsives:
c3    - la permiere manoeuvre, situee a l'apoastre, consiste a placer le
c3      periastre a l'altitude visee;
c3    - la seconde manoeuvre, situee au nouveau periastre, consiste a de
c3      placer l'apoastre pour circulariser l'orbite (cf NOTA)
c3
c3    NOTA Actuellement, on suppose l'orbite de parking circulaire.
c3
c3......................................................................
c4    variables d'entree
c4
c4    xorbit(7)         R8    parametres orbitaux
c4    xposit(3)         R8    posiiton absolue geocentrique spherique
c4    xvites(3)         R8    vitesse relative locale spherique
c4    ifinal            I4    indicateur de fin de simulation
c4......................................................................
c6    variables de sortie
c6
c6    deltav(3)         R8    couts des maneouvres propuslives elementai
c6                            res et totale
c6    dvopti(3)         R8    DV sur la trajectoire optimale
c6......................................................................
c7    variables internes
c7
c7    rapoge            R8    rayon apoastre fin aerocapture
c7    rapotf            R8    rayon apoastre orbite de parking
c7    rperig            R8    rayon periastre fin aerocapture
c7......................................................................
c8    composants appelants
c8
c8    simmsr            INT   simulation d'aerocapture
c8......................................................................
c10   commons utilises
c10
c10   geoide                  caracteristiques champ de pesanteur
c10   orbvis                  caracteristiques orbite visee
c10   planet                  caracteristiques planete
c10   parkin                  caracteristiques orbite de parking
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  ergols (xorbit,xposit,xvites,ifinal,
     +                    deltav,dvopti)
c
      implicit none
c
      integer  ifinal,
     +         i
c
      double precision  xorbit(13),xposit(3),xvites(3),deltav(4),
     +                  dvopti(4),
     +                  demiax,excent,excorb,gomega,rapotf,rapoge,
     +                  requat,rperig,rpertf,rpolar,vitfin,vitini,
     +                  xincli,xj2,xmug,xomega,zapoge,zapotf,
     +                  zperig,zpertf,vitneu(2),dincli,rayneu(2),
     +                  anoneu(2),degrad,pi
c
      common / geoide / excent,xj2,xmug
      common / orbvis / zapoge,zperig,demiax,excorb,xincli,gomega
      common / planet / xomega(3),requat,rpolar
      common / parkin / zapotf,zpertf
      common / trigon / degrad,pi
c
      intrinsic  dabs,dcos,dmin1,dsin,dsqrt
c
      if (ifinal.ne.3) then
c
c		pas de changement d'orbite
c
         do  i = 1,4
             deltav(i) = 1.d30
         end do
         
      else
c
c		apoastres, periastres...
c
         rapoge = requat + xorbit(7)
         rperig = requat + xorbit(6)
         rapotf = requat + zapotf
         rpertf = requat + zpertf
c
c		1ere manoeuvre
c
         vitfin = dsqrt(2.d0*xmug*rpertf/
     +                           (rapoge*(rapoge + rpertf)))
         vitini = dsqrt(2.d0*xmug*rperig/
     +                           (rapoge*(rapoge + rperig)))
c
         deltav(1) = vitfin - vitini
c
c		2nde manoeuvre
c
         vitfin = dsqrt(2.d0*xmug*rapotf/
     +                           (rpertf*(rapotf + rpertf)))
         vitini = dsqrt(2.d0*xmug*rapoge/
     +                           (rpertf*(rapoge + rpertf)))
c
         deltav(2) = vitfin - vitini
c
c		correction inclinaison (sur orbite finale)
c
         anoneu(1) = 2.d0*pi - xorbit(5)
         anoneu(2) =      pi - xorbit(5)
         do  i = 1,2
             rayneu(i) = demiax*(1.d0 - excorb**2)/
     +                  (1.d0 + excorb*dcos(anoneu(i)))
             vitneu(i) = dsqrt(2.d0*xmug*(1.d0/rayneu(i) - 
     +                         1.d0/(2.d0*demiax)))
         end do
         dincli    = dabs(xincli - xorbit(3))
         deltav(3) = 2.d0*dmin1(vitneu(1),vitneu(2))*dsin(dincli/2.d0)
c
c		cout global (longi + lateral)
c
         deltav(4) = dabs(deltav(1)) + dabs(deltav(2)) + 
     +               dabs(deltav(3))
c
      endif
c
c		DV optimal (parametres missions)
c
      rapoge = requat + zapoge
      rperig = requat + zperig
      rapotf = requat + zapotf
      rpertf = requat + zpertf
c
c		1ere manoeuvre
c  
      vitfin = dsqrt(2.d0*xmug*rpertf/
     +                        (rapoge*(rapoge + rpertf)))
      vitini = dsqrt(2.d0*xmug*rperig/
     +                        (rapoge*(rapoge + rperig)))
c
      dvopti(1) = vitfin - vitini
c
c		2nde manoeuvre
c
      vitfin = dsqrt(2.d0*xmug*rapotf/
     +                        (rpertf*(rapotf + rpertf)))
      vitini = dsqrt(2.d0*xmug*rapoge/
     +                        (rpertf*(rapoge + rpertf)))
c
      dvopti(2) = vitfin - vitini
c
c		cout de correction de l'inclinaison
c     
      dvopti(3) = 0.d0 
c
c		cout global longi + lateral
c
      dvopti(4) = dabs(dvopti(1)) + dabs(dvopti(2)) + 
     +            dabs(dvopti(3))

c
      return
      end
