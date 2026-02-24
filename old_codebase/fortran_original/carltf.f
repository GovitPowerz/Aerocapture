c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : carltf.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module realise la sauvegarde des conditions finales en cas de
c3    Monte-Carlo (sauvegarde sur fichier formatte a acces sequentiel).
c3
c3    NOTA  les valeurs angulaires sont sauvegardees en degrees, les al-
c3          titudes en km, les flux en kW/m2, les pressions dynamiques
c3          en kPa et les accelerations en nombre de g terrestre
c3          le signe - sur l'azimut de la vitesse est du a la convention
c3          de signe adoptee par MAXTOM
c3......................................................................
c4    variables d'entree
c4
c4    xorbit(7)         R8    parametres orbitaux
c4    xposit(3)         R8    position reelle repere geocentrique
c4    xvites(3)         R8    vitesse reelle repere local
c4    altmax(3)         R8    altitude de valeurs max
c4    datmax(3)         R8    instants de valeurs max
c4    tpcnum(3)         R8    duree max d'integration de trajectoire
c4    deltav(3)         R8    cout pour rejoindre l'orbite de parking
c4    fluter(2)         R8    flux thremique courant et max
c4    fcharg(2)         R8    facteur de charge courant et max
c4    pdynam(2)         R8    pression dynamique courante et max
c4    finess            R8    finesse equilibree
c4    somflu            R8    integrale de flux
c4    somgit            R8    gite consommee
c4    tcaptr            R8    duree de la phase de capture
c4    temsim            R8    temps courant
c4    trebon            R8    date de rebond
c4    zrebon            R8    altitude de rebond
c4    ifinal            I4    indicateur de fin de simulation
c4    iprepr            I4    compteur de commutations en preprogramme
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
c9......................................................................
c10   commons utilises
c10
c10   gravit                  accelerations de gravite
c10   orbvis                  caracteristiques orbite visee
c10   period                  cadences integration...
c10   trigon                  parametres trigonometriques
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  carltf (xorbit,ecartr,xposit,xvites,altmax,datmax,
     +                    deltav,fluter,fcharg,pdynam,finess,somflu,
     +                    somgit,tcaptr,temsim,trebon,zrebon,iprepr,
     +                    ifinal,numero,nbroll)
c
      implicit none
c
      integer  ifinal,iprepr(2),numero,nbroll,
     +         natsim,k,isucces,i,natman
c
      double precision  xorbit(13),ecartr(4),xposit(3),xvites(3),
     +                  altmax(3),datmax(3),deltav(4),fluter(2),
     +                  fcharg(2),pdynam(2),finess,somflu,somgit,
     +                  tcaptr,temsim,trebon,zrebon,
     +                  altitu,azmvit,degrad,demiax,excorb,gomega,
     +                  g0terr,g0mars,latitu,longit,penvit,pi,
     +                  rayvec,tguida,tinteg,tnavig,tpilot,tpredi,
     +                  vitess,vitrad,xenerg,xincli,xprepr(2),
     +                  xsauve(53),zapoge,zperig,epsiln,tactiv,
     +                  tsecur,errinc,errvit,errzap,errzpe,
     +                  anoinf,vitinf,
     +                  enrtot
c
      common / gravit / g0terr,g0mars
      common / misaga / anoinf,vitinf
      common / modaga / natman
      common / modgui / natsim
      common / orbvis / zapoge,zperig,demiax,excorb,xincli,gomega
      common / period / tnavig,tguida,tpilot,tpredi,tinteg
      common / trigon / degrad,pi
      common / vlimit / epsiln
      common / succes / errinc,errvit,errzap,errzpe
c
      intrinsic  dble,dsin,dmax1,dmin1
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
      if (natsim.eq.2) then
         tcaptr = temsim + 1.d-7
      endif
      if (natsim.eq.3) then
         if (tcaptr.le.epsiln) tcaptr = 1.d-7
      endif 
      xprepr(1) = tguida*dble(iprepr(1))/temsim
      xprepr(2) = tguida*dble(iprepr(2))/temsim
c            
      tactiv = temsim - tguida*dble(iprepr(2))
      tsecur = tguida*dble(iprepr(1))
c
c		securiastion (chronologie a revoir...pour 1 pas d'integration)
c      
      if (tsecur.ge.temsim) then
         tsecur = temsim
      endif
      if (tactiv.le.0.d0) then
         tactiv = 0.d0
      endif
      do  i = 1,2
          if (xprepr(i).ge.100.d0) then
             xprepr(i) = 100.d0
          endif
      end do
c
c		determination altitude
c
      call  frayon (xposit,
     +              altitu,latitu)
c
c		parametres energetiques
c
      vitrad = vitess*dsin(penvit)
      xenerg = enrtot (xposit,xvites)
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
c
      xsauve(8) = vitrad
      xsauve(9) = xenerg/1.d6
c
      xsauve(10) = xorbit(1)/1.d3
      xsauve(11) = xorbit(2)
      xsauve(12) = xorbit(3)/degrad
      xsauve(13) = xorbit(4)/degrad
      xsauve(14) = xorbit(5)/degrad
      xsauve(15) = xorbit(8)/degrad
      xsauve(16) = xorbit(6)/1.d3
      xsauve(17) = xorbit(7)/1.d3
c
      xsauve(18) = fluter(2)/1.d3
      xsauve(19) = fcharg(2)/g0terr
      xsauve(20) = pdynam(2)/1.d3
      xsauve(21) = altmax(1)/1.d3
      xsauve(22) = altmax(2)/1.d3
      xsauve(23) = altmax(3)/1.d3
      xsauve(24) = datmax(1)
      xsauve(25) = datmax(2)
      xsauve(26) = datmax(3)
c
      xsauve(27) = zrebon/1.d3
      xsauve(28) = trebon
      xsauve(29) = temsim
      xsauve(30) = somflu/1.d6
      xsauve(31) = xsauve(16) - zperig/1.d3
      xsauve(32) = xsauve(17) - zapoge/1.d3
      xsauve(33) = ifinal
      xsauve(34) = ecartr(1)/1.d3
      xsauve(35) = ecartr(2)
      xsauve(36) = ecartr(3)/degrad
      xsauve(37) = ecartr(4)/degrad
      xsauve(38) = finess
      xsauve(39) = deltav(1)
      xsauve(40) = deltav(2)
      xsauve(41) = deltav(3)
      xsauve(42) = abs(deltav(1)) + dabs(deltav(2))
      xsauve(43) = deltav(4)
      xsauve(44) = dmin1(xprepr(1)*100.d0,100.d0)
      xsauve(45) = dmin1(xprepr(2)*100.d0,100.d0)
      xsauve(46) = dmin1(100.d0*tsecur/tactiv,100.d0)
      xsauve(47) = somgit/degrad
      xsauve(48) = xorbit(9)
      xsauve(49) = xorbit(10)/degrad
      xsauve(50) = dble(nbroll)
      xsauve(51) = xorbit(9) - vitinf
      xsauve(52) =(xorbit(10) - anoinf)/degrad
c
c		respect des exigences GNC finales
c
      isucces = 0
      
      if (dabs(xsauve(32)).le.errzap) then
         isucces = isucces + 1
      endif
      if (dabs(xsauve(31)).le.errzpe) then
         isucces = isucces + 3
      endif
      if (dabs(xsauve(36)).le.errinc) then
         isucces = isucces + 5
      endif
      if (dabs(xsauve(43)).le.errvit) then
         isucces = isucces + 7
      endif            
      
      xsauve(53) = dble(isucces)
c
      write(310,1000) numero,(xsauve(k), k = 2,53)
c
c		modification du fichier rempli au module carltz
c
      backspace (unit= 300)
      read (300,2000) numero,(xsauve(k), k = 2,36)
      backspace(unit= 300)
      xsauve(35) = dble(isucces)
      write(300,2000) numero,(xsauve(k), k = 2,36)
c
 1000 format(1x,i5,52(1x,d15.7))
 2000 format(1x,i5,35(1x,d17.5))
c
      return
      end
