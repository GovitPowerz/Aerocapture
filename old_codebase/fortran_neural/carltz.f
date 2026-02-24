c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : carltz.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module realise la sauvegarde des conditions initiales en cas de
c3    Monte-Carlo (sauvegarde sur fichier formatte a acces sequentiel).
c3
c3    NOTA  le signe - sur l'azimut de la vitesse est du a la convention
c3          de signe adoptee par MAXTOM
c3......................................................................
c4    variables d'entree
c4
c4    xposit(3)         R8    position reelle repere geocentrique
c4    xvites(3)         R8    vitesse reelle repere local
c4    numero            I4    numero de simulation courant
c4......................................................................
c7    variables internes
c7
c7    altitu            R8    altitude courante
c7    latitu            R8    latitude courante
c7    longit            R8    longitude courante
c7    rayvec            R8    rayon vecteur courant
c7    vitess            R8    norme de la vitesse relative
c7    vitrad            R8    vitesse radiale
c7    xenerg            R8    energie totale
c7    xrayon            R8    rayon planete courant
c7......................................................................
c8    composants appelants
c8
c8    simmsr            INT   simulation aerocapture
c8......................................................................
c9    composants appeles
c9
c9    frayon            INT   rayon planete
c9    enrtot            INT   abscisse du profil de gite commandee
c9......................................................................
c10   commons utilises
c10
c10   geoide                  caracteristqiues champ de pesanteur
c10   trigon                  parametres trigonometriques
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  carltz (xposit,xvites,numero)
c
      implicit none
c
      integer  numero,k
c
      double precision  xposit(3),xvites(3),
     +                  altitu,azmvit,cxenom,czenom,dalfae,degrad,
     +                  disatm,dispos,disvit,dxdrag,dxlift,dxposi,
     +                  dxvite,findis,finnom,latitu,longit,penvit,pi,
     +                  positz,rayvec,vitess,vitesz,vitrad,xenerg,
     +                  xorbit(13),xorbiz(7),xsauve(36),alfaeq,dadrag,
     +                  dnlift,dxmass,
     +                  enrtot
c
      common / aeroeq / alfaeq
      common / aernom / cxenom,czenom,finnom
      common / mecaer / dalfae,disatm,dadrag,dnlift
      common / mecmas / dxmass
      common / perinj / dxposi(3),dxvite(3)
      common / pernav / dispos(3),disvit(3)
      common / trigon / degrad,pi
      common / xvrent / positz(3),vitesz(3)
c
      intrinsic  dsin
c
      external  enrtot
c
      rayvec = xposit(1)
      longit = xposit(2)
      latitu = xposit(3)
      vitess = xvites(1)
      penvit = xvites(2)
      azmvit = xvites(3)
c
c		determination altitude
c
      call  frayon (xposit,
     +              altitu,latitu)
c
c		parametres orbitaux
c
      call  orbito (xposit,xvites,
     +              xorbit)
c
c		parametres energetiques
c
      vitrad = vitess*dsin(penvit)
      xenerg = enrtot (xposit,xvites)
      
c
c		parametres aerodynamiques
c
      dxdrag = dadrag*dcos(alfaeq + dalfae) + 
     +         dnlift*dsin(alfaeq + dalfae)
      dxlift =-dadrag*dsin(alfaeq + dalfae) + 
     +         dnlift*dcos(alfaeq + dalfae)
     
      findis = (czenom*(1.d0 + dxlift))/
     +         (cxenom*(1.d0 + dxdrag))     
c
c		sauvegarde
c
      xsauve(1) = numero
      xsauve(2) = altitu/1.d3
      xsauve(3) = longit/degrad
      xsauve(4) = latitu/degrad
      xsauve(5) = vitess
      xsauve(6) = penvit/degrad
      xsauve(7) = azmvit/degrad
      xsauve(8) = vitrad
      xsauve(9) = xenerg/1.d6
c
      xsauve(10) = disatm*100.d0
      xsauve(11) = dxdrag*100.d0
      xsauve(12) = dxlift*100.d0
      xsauve(13) =(findis/finnom - 1.d0)*100.d0
      xsauve(14) = dxposi(1)/1.d3
      xsauve(15) = dxposi(2)/degrad
      xsauve(16) = dxposi(3)/degrad
      xsauve(17) = dxvite(1)
      xsauve(18) = dxvite(2)/degrad
      xsauve(19) = dxvite(3)/degrad
      xsauve(20) = dispos(1)/1.d3
      xsauve(21) = dispos(2)/degrad
      xsauve(22) = dispos(3)/degrad
      xsauve(23) = disvit(1)
      xsauve(24) = disvit(2)/degrad
      xsauve(25) = disvit(3)/degrad
c
      xsauve(26) = xorbit(2)
      xsauve(27) = xorbit(3)/degrad
      xsauve(28) = xorbit(6)/1.d3
      xsauve(29) = xorbit(7)/1.d3
c
      xsauve(26) = xorbit(2)
      xsauve(27) = xorbit(3)/degrad
      xsauve(28) = xorbit(6)/1.d3
      xsauve(29) = xorbit(7)/1.d3
c
c		parametres orbitaux initiaux nominaux
c
      call  orbito (positz,vitesz,
     +              xorbiz)
c
      xsauve(30) = xorbit(2) - xorbiz(2)
      xsauve(31) =(xorbit(3) - xorbiz(3))/degrad
      xsauve(32) =(xorbit(6) - xorbiz(6))/1.d3
      xsauve(33) =(xorbit(7) - xorbiz(7))/1.d3 
           
      xsauve(34) = dalfae/degrad
      xsauve(35) = 0.d0
      xsauve(36) = dxmass*100.d0
c
      write(300,1000) numero,(xsauve(k), k = 2,36)
c
 1000 format(1x,i5,35(1x,d17.5))
c
      return
      end
