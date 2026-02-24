c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : orbito.f
c2    date   : 12/07/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine les parametres orbitaux a partir de la positi-
c3    on et de la vitesse courantes de la capsule, a savoir:
c3    - longitude du noeud ascendant;
c3    - inclinaison;
c3    - demi grand-axe;
c3    - argument du periastre;
c3    - anomalie vraie;
c3    - rayons du periastre et de l'apoastre (et les altitudes equatoria
c3      les associees)
c3
c3    NOTA  - La position de l'engin doit etre exprimee dans un repere
c3            centre sur la planete;
c3          - La vitesse de l'engin doit etre exprimee dans un repere
c3            equatorial galileen
c3          - pour le calcul du demi grand axe a, on tient compte d'un
c3            seuil sur l'energie pour eviter la singularite au passage
c3            par la parabole (cas E = 0)
c3......................................................................
c4    variables d'entree
c4
c4    positx(3)         R8    position absolue                       (m)
c4    vitesx(3)         R8    vitesse relative                     (m/s)
c4......................................................................
c6    variables de sortie
c6
c6    anomal            R8    anomalie vraie                       (rad)
c6    demiax            R8    demi grand-axe                         (m)
c6    excent            R8    excentricite
c6    gomega            R8    longitude du noeud ascendant         (rad)
c6    pomega            R8    argument du periastre                (rad)
c6    rayapo            R8    rayon vecteur apoastre                 (m)
c6    rayper            R8    rayon vecteur periastre                (m)
c6    xincli            R8    inclinaison                          (rad)
c6......................................................................
c7    variables internes
c7
c7    enrorb            R8    energie totale
c7    posita(3)         R8    posiiton absolue
c7    vitabs            R8    norme de la vitesse absolue          (m/s)
c7    vitesa(3)         R8    vitesse absolue repere geocentrique
c7    xcinet            R8    norme du moment cinetique
c7    xmocin(3)         R8    moment cinetique
c7    xmug              R8    constante gravitaionnelle
c7......................................................................
c8    composants appelants
c8
c8    matcon            INT   determination et gestion des contraintes
c8    realit            INT   integration trajectoire reelle
c8......................................................................
c9    composants appeles
c9
c9    pnorme            INT   norme d'un vecteur
c9    pscalr            INT   produit scalaire de 2 vecteurs
c9    pvecto            INT   produit vectoriel
c9    xvabsl            INT   position et vitesse absolue
c9......................................................................
c10   commons utilises
c10
c10   geoide                  caracteristiques champ de pesanteur
c10   planet                  caracteristqiues planete
c10   satorb                  seuil energetique pour calcul de a
c10   trigon                  constantes trigonometriques
c10   vlimit                  seuil de comparaison
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  orbito  (xposit,xvites,
     +                     xorbit)
c
      implicit none
c
      double precision  xposit(3),xvites(3),xorbit(13),
     +                  cosarg,cosinc,cosomg,degrad,demiax,enrmin,
     +                  enrorb,epsiln,excent,exentr,gomega,parexc,
     +                  paromg(4),xmocin(3),pi,pomega,posita(3),
     +                  posvit,rayapo,rayper,rayvec,requat,rpolar,
     +                  sigenr,sinarg,sininc,sinomg,vitabs,vitesa(3),
     +                  xcinet,xincli,xj2,xmug,xomega,
     +                  pnorme,pscalr
c

      double precision   sv0, cv0, v0
      double precision   vinfini,nu,vexc(3),vinf(3),vxin(3)
      integer*4 i
      
      common / geoide / exentr,xj2,xmug
      common / planet / xomega(3),requat,rpolar
      common / satorb / enrmin
      common / trigon / degrad,pi
      common / vlimit / epsiln
c
      intrinsic  dabs,datan2,dcos,dmax1,dsign,dsin,dsqrt
c
      external   pnorme,pscalr
c
c		position et vitesse absolue geocentriques cartesiennes
c
      call  xvabsl (xposit,xvites,
     +              posita,vitesa)
c
c		calculs preliminaires
c
      call  pvecto (posita,vitesa,
     +              xmocin)
c
      rayvec = pnorme (posita)
      vitabs = pnorme (vitesa)
      xcinet = pnorme (xmocin)
      posvit = pscalr (posita,vitesa)
c
c		energie totale (seuillee eventuellement)
c
      enrorb = vitabs**2/2.d0 - xmug/rayvec
      sigenr = dsign(1.d0,enrorb)
      enrorb = sigenr*dmax1(dabs(enrorb),enrmin)
c
c		demi grand axe
c
      demiax =-xmug/(2.d0*enrorb)
c
c		excentricite
c
      parexc = xcinet**2/
     +        (xmug*demiax)
c
      if (dabs(parexc - 1.d0).lt.epsiln**2) then
         excent = 0.d0
      else
         excent = dsqrt(1.d0 - parexc)
      endif
c
c		inclinaison
c
      cosinc = xmocin(3)/xcinet
      sininc = dsqrt(1.d0 - cosinc**2)
c
      xincli = datan2(sininc,
     +                cosinc)
c
c		longitude du noeud ascendant (ou ascension droite)
c
      sinomg = xmocin(1)/(xcinet*sininc)
      cosomg =-xmocin(2)/(xcinet*sininc)
      gomega = datan2(sinomg,
     +                cosomg)
c       
c		anomalie vraie (nu)
c         
      if ( enrorb.lt.0. ) then
c
c		orbite elliptique
c
	 sv0 = (posita(1)*vitesa(1) + posita(2)*vitesa(2) +
     +	        posita(3)*vitesa(3))*
     +         dsqrt(1. - excent**2)/
     +         (excent*dsqrt(xmug*demiax))
         cv0 = (1. - rayvec/demiax )/excent - excent
         v0  = datan2(sv0,cv0)
      
      else
c
c
c		orbite hyperbolique
c
         sv0 =(posita(1)*vitesa(1) + posita(2)*vitesa(2) +
     +          posita(3)*vitesa(3) )*
     +         dsqrt(excent**2 - 1. )/
     +         (excent*dsqrt(xmug*dabs(demiax)))
         cv0 =-((1.d0 + rayvec/dabs(demiax))/excent - excent )
         v0  = datan2(sv0,cv0 )
      endif 	 
c
c		argument periastre
c
      if ( xincli.gt.1.d-3 ) then
         pomega = datan2(posita(3)/(dsin(xincli)*rayvec),
     +                  (posita(1)*cosomg + posita(2)*sinomg)/rayvec) -
     +            v0
      else
         pomega = datan2(posita(2),posita(1)) -
     +            v0
      endif
      if (pomega.lt.0.) then
         pomega = 2.d0*pi + pomega 
      endif
c
c		periastre
c
      rayper = demiax*(1.d0 - excent)
c
c		apoastre
c
      rayapo = demiax*(1.d0 + excent)
c
c		vitesse infinie dans le cas d'une hyperbole
c
      if (enrorb.gt.0.) then
	 vinfini = dsqrt(-xmug/demiax)
	 nu      = dasin(1/excent)
	 
         call  pvecto (vitesa,xmocin,
     +                 vexc)
	 do  i = 1,3
	     vexc(i) =(vexc(i)/xmug) - (posita(i)/rayvec)
	     vexc(i) = vexc(i)/excent
	 end do
         
	 call  pvecto (vexc,xmocin,
     +                 vxin)
	 do  i = 1,3
	     vxin(i) = vxin(i)/xcinet
	 end do
	 
	 do  i = 1,3
	     vinf(i) = vinfini*(-dsin(nu)*vexc(i) - dcos(nu)*vxin(i))
	 end do
	    
      endif 	 
c
      xorbit(1) = demiax
      xorbit(2) = excent
      xorbit(3) = xincli
      xorbit(4) = gomega
      xorbit(5) = pomega
      xorbit(6) = rayper - requat
      xorbit(7) = rayapo - requat
      xorbit(8) = v0
      
      if (enrorb.gt.0.) then
         xorbit(9)  = vinfini
         xorbit(10) = nu
         xorbit(11) = vinf(1)
         xorbit(12) = vinf(2)
         xorbit(13) = vinf(3)
      else
         xorbit(9) = 0.
         xorbit(10) = 0.
         xorbit(11) = 0.
         xorbit(12) = 0.
         xorbit(13) = 0.
      endif
c
      return
      end
