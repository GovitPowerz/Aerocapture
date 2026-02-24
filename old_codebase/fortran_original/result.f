c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : result.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module realise la sauvegarde des parametres de vol courant
c3
c3    NOTA  les valeurs angulaires sont sauvegardees en degres, les alti
c3          tudes en km, les accelerations en nombre de g terrestre, les
c3          flux en kW/m2, les pressions dynamqiues en kPa
c3          le signe - sur l'azimut de la vitesse reprend la convention
c3          de signe adoptee sur MAXTOM pour la comparaison de la trajec
c3          toire guidee avec la trajectoire optimale.
c3......................................................................
c4    variables d'entree
c4
c4    xorbit(7)         R8    parametres orbitaux
c4    ecartn(4)         R8    ecart final predit sur les contraintes
c4    ecartr(4)         R8    ecart courant sur les contraintes finales
c4    positr(3)         R8    position absolue goecentrqiue reelle
c4    vitesr(3)         R8    vitesse relative locale reelle
c4    positn(3)         R8    position absolue geocentrique estimee
c4    vitesn(3)         R8    vitesse relative locale estimee
c4    tpcnum(3)         R8    duree max d'integration
c4    acceln(2)         R8    accelerations aerodynamiques estimees
c4    accelr(2)         R8    accelerations aerodynamiques reelles
c4    fluter(2)         R8    flux thermique courant et max
c4    fcharg(2)         R8    facteur de charge courant et max
c4    pdynam(2)         R8    pression dynamique courante et max
c4    coefro            R8    coeffcient d'estimation de la densite
c4    energr            R8    energie totale
c4    gitcom            R8    gite comamndee
c4    gitpil            R8    gite realisee
c4    roexit            R8    densite atmospherique finale predite
c4    roguid            R8    densite atmospherique courante estimee
c4    romver            R8    masse volumique de l'air reelle
c4    somflu            R8    integrale de flux
c4    temsim            R8    temps courant
c4    vitgit            R8    vitesse de gite avant saturation
c4    vitmac            R8    nombre de Mach
c4    tpctra            R8    duree predite de trajectoire
c4    isatur            I4    indicateur de saturation de vitesse de gite
c4    isecur            I4    indicateur de securisation du guidage
c4......................................................................
c7    variables internes
c7
c7    altitu            R8    alittude
c7    latitu            R8    latitude
c7    penvit            R8    pente vitesse relative
c7    rayvec            R8    altitude geocentrique
c7    vitess            R8    vitesse relative
c7    vitrad            R8    vitesse radiale
c7    xenerg            R8    energie
c7......................................................................
c8    composants appelants
c8
c8    simmsr            INT   simulation d'aerocapture
c8......................................................................
c9    composants appeles
c9
c9    frayon            INT  calcul rayon planete cible
c9......................................................................
c10   commons utilises
c10
c10   geoide                 caracteristiques champ de pesanteur
c10   gravit                 accelerations de pesanteur
c10   trigon                 constantes trigonometriques
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  result (xorbit,ecartr,ecartn,positr,vitesr,
     +                    positn,vitesn,acceln,accelr,dzapog,
     +                    fluter,fcharg,pdynam,alfcom,alfpil,
     +                    coefro,energr,gitcom,gitpil,roexit,
     +                    roguid,romver,somflu,temsim,tpctra,
     +                    vitgit,vitmac,vitref,iguida,ilongi,
     +                    indext,indrol,isatur,isecur)
c
      implicit none
c
      include '../include/dimensions.incl'
c
      integer isatur,isecur,k,indext,indrol,iguida(2),ilongi,
     +        icmlon,icmlat
c
      double precision  xorbit(13),ecartr(4),ecartn(4),positr(3),
     +                  vitesr(3),positn(3),vitesn(3),acceln(2),
     +                  accelr(2),dzapog(2),fluter(2),fcharg(2),
     +                  pdynam(2),alfcom,alfpil,coefro,energr,
     +                  gitcom,gitpil,roexit,roguid,romver,somflu,
     +                  temsim,tpcnum(3),tpctra,vitgit,vitmac,vitref,
     +                  altitu,azmvit,cstgam,degrad,excent,facech,
     +                  g0terr,g0mars,gaindh,gainpd,latitu,longit,
     +                  penvit,pdyneq,pi,rayvec,rmoyen,rozmod,srefer,
     +                  vgitmx,vitess,vitrad,xj2,xmasse,xmug,
     +                  xsauve(74),zromod,enrlat,pdacti,pdinib,
     +                  coridx,coridy
c
      common / capsul / srefer,vgitmx,xmasse
      common / congui / pdacti,pdinib
      common / corrid / coridx,coridy
      common / geoide / excent,xj2,xmug
      common / gravit / g0terr,g0mars
      common / loglat / enrlat(2)
      common / modatm / cstgam,facech,rozmod,rmoyen,zromod
      common / trigon / degrad,pi
c
      common / pargui / pdyneq
      common / gains / gaindh,gainpd
c
      intrinsic  dble,dexp,dsin
c
      if ((energr.le.pdacti).and.(energr.ge.pdinib)) then
         icmlon = 1
      else
         icmlon = 0
      endif
      if ((energr.le.enrlat(1)).and.(energr.ge.enrlat(2))) then
         icmlat = 1
      else
         icmlat = 0
      endif 
c
c		parametres reels
c
      rayvec = positr(1)
      longit = positr(2)
      latitu = positr(3)
      vitess = vitesr(1)
      penvit = vitesr(2)
      azmvit = vitesr(3)
c
c		determination altitude geodesique
c
      call  frayon (positr,
     +              altitu,latitu)
c
c		parametres energetiques
c
      vitrad = vitess*dsin(penvit)
c
      xsauve(1) = temsim
      xsauve(2) = altitu/1.d3
      xsauve(3) = longit/degrad
      xsauve(4) = latitu/degrad
      xsauve(5) = vitess
      xsauve(6) = penvit/degrad
      xsauve(7) = azmvit/degrad
c
      xsauve(8) = vitrad
      xsauve(9) = energr/1.d6
c
      xsauve(10) = xorbit(1)/1.d3
      xsauve(11) = xorbit(2)
      xsauve(12) = xorbit(3)/degrad
      xsauve(13) = xorbit(4)/degrad
      xsauve(14) = xorbit(5)/degrad
      xsauve(15) = xorbit(6)/1.d3
      xsauve(16) = xorbit(7)/1.d3
      xsauve(17) = isatur
c
      xsauve(18) = romver
      xsauve(19) = roguid
      xsauve(20) = roexit
c
      xsauve(21) =(positr(1) - positn(1))/1.d3
      xsauve(22) =(positr(2) - positn(2))/degrad
      xsauve(23) =(positr(3) - positn(3))/degrad
      xsauve(24) = vitesr(1) - vitesn(1)
      xsauve(25) =(vitesr(2) - vitesn(2))/degrad
      xsauve(26) =(vitesr(3) - vitesn(3))/degrad
c
      xsauve(27) = alfpil/degrad
      xsauve(28) = alfcom/degrad
c
      write(201,1001) (xsauve(k), k = 1,28)
c
      xsauve(1)  = temsim
      xsauve(2)  = fluter(1)/1.d3
      xsauve(3)  = fcharg(1)/g0terr
      xsauve(4)  = pdynam(1)/1.d3
      xsauve(5)  = accelr(1)/g0terr
      xsauve(6)  = accelr(2)/g0terr
      xsauve(7)  = gitcom/degrad
      xsauve(8)  = gitpil/degrad
      xsauve(9)  = vitgit/degrad
      xsauve(10) = vitmac
      xsauve(11) = acceln(1)/g0terr
      xsauve(12) = acceln(2)/g0terr
      xsauve(13) = romver
      xsauve(14) = somflu/1.d6
      xsauve(15) = vitrad
      xsauve(16) = isatur
      xsauve(17) = roguid
      xsauve(18) = coefro
      xsauve(19) = isecur
      xsauve(20) = altitu/1.d3
      xsauve(21) = vitrad
      xsauve(22) = energr/1.d6
      xsauve(23) = alfcom/degrad
      xsauve(24) = alfpil/degrad
c
      write(202,1002) (xsauve(k), k = 1,24)
c
      xsauve(1)  = temsim
      xsauve(2)  = ecartn(1)/1.d3
      xsauve(3)  = ecartn(2)
      xsauve(4)  = ecartn(3)/degrad
      xsauve(5)  = ecartn(4)/degrad
      xsauve(6)  = ecartr(1)/1.d3
      xsauve(7)  = ecartr(2)
      xsauve(8)  = ecartr(3)/degrad
      xsauve(9)  = ecartr(4)/degrad
      xsauve(10) = tpcnum(1)
      xsauve(11) = tpcnum(2)
      xsauve(12) = tpcnum(3)
      xsauve(13) = tpctra
      xsauve(14) = isatur
      xsauve(15) = romver
      xsauve(16) = rozmod*dexp(-facech*(altitu - zromod))
      xsauve(17) = altitu/1.d3
      xsauve(18) = vitrad
      xsauve(19) = energr/1.d6
      xsauve(20) = pdyneq/1.d3
      xsauve(21) = pdynam(1)
      xsauve(22) = pdyneq
      xsauve(23) = indrol
      xsauve(24) = gaindh
      xsauve(25) = gainpd
      xsauve(26) = indext
      xsauve(27) = vitref
      xsauve(28) = dzapog(1)/1.d3
      xsauve(29) = iguida(1)*icmlon
      xsauve(30) = iguida(2)*icmlat
      xsauve(31) = vitesn(1)
      xsauve(32) = xorbit(3)/degrad
      xsauve(33) = dzapog(2)/1.d3
      xsauve(34) = ilongi
      xsauve(35) =((vitesr(1)/coridx)**4 + coridy)/degrad
      xsauve(36) = vitesr(1)
c
      if (indrol.eq.1) then
         indrol = 0
      endif
c
      write(203,1003) (xsauve(k), k = 1,36)


      if (indext.eq.1) then
         indext = 0
      endif
c
c		valeurs propres (ncont < 6) et profil (nsegmx < 10)
c
      xsauve(1) = temsim
      xsauve(2) = isatur
      xsauve(3) = 0
      xsauve(4) = 0
      xsauve(5) = 0
      do  k = 1,64
          xsauve(5+k) = 0.d0
      end do
c
      write(204,1004) (xsauve(k), k = 1,69)
c
 1001 format(28(1x,d20.10))
 1002 format(24(1x,d20.10))
 1003 format(36(1x,d20.10))
 1004 format(69(1x,d20.10))
 1005 format(3(1x,1pe23.16))
c
      return
      end
      
