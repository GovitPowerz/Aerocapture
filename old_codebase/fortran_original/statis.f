c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : statis.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module realise une analyse statistique simplifiee (calcul de
c3    moyenne, ecart-type et valeurs maximales) des resultats de simula
c3    tion en cas d'utilisation en Monte-Carlo
c3
c3    NOTA  pour l'etude statistique, on ne tient compte que des condi-
c3          tions fnales obtenues pour une sortie d'atmosphere sur crite
c3          re d'altitude.
c3......................................................................
c4    variables d'entree
c4
c4    nbsimu            I4    nombre de simulations
c4......................................................................
c7    variables internes
c7
c7    iechec            I4    nombre d'echecs du guidage en aerocapture
c7    irobit            I4    nombre d'orbites non viables
c7    isucce            I4    nombre de sorties d'atmopshere correctes
c7......................................................................
c8    composants appelants
c8
c8    captur            INT   programme principal aerocapture
c8......................................................................
c9    composants appeles
c9
c9    ecrtyp            INT   calcul ecart-type arithmetique
c9    maxmax            INT   recherche des valeurs maximales
c9    moyenn            INT   calcul moyenne arithmetique
c9......................................................................
c10   commons utilises
c10
c10   ficres                  suffixe des fichiers resultats
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  statis (nbsimu)
c
      implicit none
c
      integer  nbsauv
      parameter (nbsauv = 53)
c
      integer  nbsimu,
     +         i,iechec,ielipt,ihyper,indreb,iorbit(2),irebon,isimul,
     +         isorti,isucce,j,k,numero,nummax(nbsauv,2,2),
     +         nummin(nbsauv,2),numsuc(5),natman,isucces
c
      double precision  errinc,errvit,errzap,errzpe,
     +                  valmax(nbsauv,2,2),valmin(nbsauv,2),
     +                  xcarlo(nbsauv),xcarmx(nbsauv,2),
     +                  xcarmin(nbsauv,2),xectyp(nbsauv,2),
     +                  xmoyen(nbsauv,2),xtotcr(nbsauv,2),
     +                  xtotmo(nbsauv,2),xtotal(nbsauv,2)
c
      character *72  sufres
      character *23  variab(nbsauv)
      character *5   unites(nbsauv)
c
      common / ficres / sufres
      common / modaga / natman
      common / succes / errinc,errvit,errzap,errzpe
c
      intrinsic  dabs,dble,dmax1,dmin1,dsqrt
c
      iechec = 0
      irebon = 0
      isucce = 0

      ielipt = 0
      ihyper = 0
      isorti = 0
      iorbit(1) = 0
      iorbit(2) = 0
      do  i = 1,5
          numsuc(i) = 0
      end do
c
      variab(1)  = '   '
      variab(2)  = 'altitude '
      variab(3)  = 'longitude '
      variab(4)  = 'latitude'
      variab(5)  = 'relative velocity'
      variab(6)  = 'flight path angle'
      variab(7)  = 'heading angle'
      variab(8)  = 'vertical velocity'
      variab(9)  = 'total energy'
      variab(10) = 'semi major axis a'
      variab(11) = 'eccentricity e'
      variab(12) = 'inclination i'
      variab(13) = 'longitude W'
      variab(14) = 'periapsis argument'
      variab(15) = 'true anomaly'
      variab(16) = 'Z periapsis'
      variab(17) = 'Z apoapsis'
      variab(18) = 'thermal flux'
      variab(19) = 'load factor'
      variab(20) = 'dynamic pressure'
      variab(21) = 'Z flux max'
      variab(22) = 'Z gamma max'
      variab(23) = 'Z Pdyn max'
      variab(24) = 'T flux max'
      variab(25) = 'T gamma max'
      variab(26) = 'T Pdyn max'
      variab(27) = 'Z skipping'
      variab(28) = 'T skipping'
      variab(29) = 'maneuver duration'
      variab(30) = 'heat load'
      variab(31) = 'periapsis offset Zp'
      variab(32) = 'apoapsis offset Za'
      variab(33) = ' '
      variab(34) = 'semi major axis offset a'
      variab(35) = 'eccentricity offset e'
      variab(36) = 'inclination offset i '
      variab(37) = 'ascending node offset W'
      variab(38) = 'L/D ratio'
      variab(39) = 'cost DV 1 periapsis'
      variab(40) = 'cost DV 2 apoapsis'
      variab(41) = 'cost DV3 inclination'
      variab(42) = 'cost DV1 + DV2 '
      variab(43) = 'cost DV1 + DV2 + DV3'
      variab(44) = 'cosinus securisation'
      variab(45) = 'guidance inhibition '
      variab(46) = 'guidance securisation '
      variab(47) = 'bank angle consumption '
      variab(48) = 'infinite velocity '
      variab(49) = 'infinite true anomaly '
      variab(50) = 'number of roll reversals'
      variab(51) = 'V inf offset'
      variab(52) = 'Nu inf offset'
      variab(53) = 'Za Zp i V fulfilment'
c
      unites(1)  = ' '
      unites(2)  = 'km'
      unites(3)  = 'deg'
      unites(4)  = 'deg'
      unites(5)  = 'm/s'
      unites(6)  = 'deg'
      unites(7)  = 'deg'
      unites(8)  = 'm/s'
      unites(9)  = 'MJ/kg'
      unites(10) = 'km'
      unites(11) = '  '
      unites(12) = 'deg'
      unites(13) = 'deg'
      unites(14) = 'deg'
      unites(15) = 'deg'
      unites(16) = 'km'
      unites(17) = 'km'
      unites(18) = 'kW/m2'
      unites(19) = 'g'
      unites(20) = 'kPa'
      unites(21) = 'km'
      unites(22) = 'km'
      unites(23) = 'km'
      unites(24) = 's'
      unites(25) = 's'
      unites(26) = 's'
      unites(27) = 'km'
      unites(28) = 's'
      unites(29) = 's'
      unites(30) = 'MJ/m2'
      unites(31) = 'km'
      unites(32) = 'km'
      unites(33) = ' '
      unites(34) = 'km'
      unites(35) = ' '
      unites(36) = 'deg'
      unites(37) = 'deg'
      unites(38) = '  '
      unites(39) = 'm/s'
      unites(40) = 'm/s'
      unites(41) = 'm/s'
      unites(42) = 'm/s'
      unites(43) = 'm/s'
      unites(44) = '%'
      unites(45) = '%'
      unites(46) = '%'
      unites(47) = 'deg'
      unites(48) = 'm/s'
      unites(49) = 'deg'
      unites(50) = ' '
      unites(51) = 'm/s'
      unites(52) = 'deg'
      unites(53) = ' ' 
c
c		initialisation
c
      do  i = 1,nbsauv
          do  j = 1,2
             nummax(i,1,j) = 1
             nummax(i,2,j) = 1
             nummin(i,j)   = 1
             xtotal(i,j)   = 0.d0
             xtotcr(i,j)   = 0.d0
             xectyp(i,j)   = 0.d0
             valmax(i,1,j) =-1.d30
             valmax(i,2,j) =-1.d30
             valmin(i,j)   = 1.d30
             xcarmx(i,j)   = 0.d0
             xmoyen(i,j)   = 0.d0
          end do
      end do
c
      do  isimul = 1,nbsimu
c
c		lecture des fichiers resultats
c
	  
	  read(310,1000) numero,(xcarlo(k), k = 2,nbsauv)
          
c
          if (xcarlo(33).ge.dble(3)) then
c
c		decompte orbites non viables sortie atmopshere
c
             if (xcarlo(16).le.0.d0) then
                iorbit(1) = iorbit(1) + 1
                if (xcarlo(15).lt.180.d0) then
                   iorbit(2) = iorbit(2) + 1
                endif
             endif
             indreb = 0
             if (xcarlo(27).lt.130.d3) then
                irebon = irebon + 1
                indreb = 1
             endif
             if (indreb.eq.1) then
                if (xcarlo(11).gt.1.d0) then
                   ihyper = ihyper + 1
                else
                   ielipt = ielipt + 1
                endif
             endif
c
             if ((xcarlo(11).le.1.d0).and.(natman.eq.1)) then
                isucce = isucce + 1
             endif
             
             if ((xcarlo(11).ge.1.d0).and.(natman.eq.2)) then
                isucce = isucce + 1
             endif
c
             if (((xcarlo(11).le.1.d0).and.(natman.eq.1)).or.
     +           ((xcarlo(11).ge.1.d0).and.(natman.eq.2))) then
c
c			decompte des cas de compatibilites finales
c
                if (dabs(xcarlo(36)).le.errinc) then
                   numsuc(1) = numsuc(1) + 1
                endif
                if (xcarlo(43).le.errvit) then
                   numsuc(2) = numsuc(2) + 1
                endif 
                if (dabs(xcarlo(32)).le.errzap) then
                   numsuc(3) = numsuc(3) + 1
                endif 
                if (dabs(xcarlo(31)).le.errzpe) then
                   numsuc(4) = numsuc(4) + 1
                endif
                if ((dabs(xcarlo(36)).le.errinc).and.
     +              (dabs(xcarlo(41)).le.errvit).and.
     +              (dabs(xcarlo(32)).le.errzap).and.                               
     +              (dabs(xcarlo(31)).le.errzpe)) then
                   numsuc(5) = numsuc(5) + 1
                endif                               
                               
                do  k = 2,nbsauv
c
c			calcul moyenne arithmetique
c
                    if (k.ne.33) then
                       xtotal(k,1) = xtotal(k,1) + xcarlo(k)
		       xmoyen(k,1) = xtotal(k,1)/dble(isucce)
c
c			calcul ecart-type arithmetique
c
                       xtotcr(k,1) = xtotcr(k,1) + xcarlo(k)**2
                       xtotmo(k,1) = xtotcr(k,1)/dble(isucce)
                       xectyp(k,1) = dsqrt(dabs(xtotmo(k,1) -
     +                                          xmoyen(k,1)**2))
c
c			recherche des valeurs maximales
c
                       xcarmx(k,1) = dmax1(xcarlo(k),valmax(k,1,1))
                       if (xcarmx(k,1).gt.valmax(k,1,1)) then
                          nummax(k,1,1) = isimul
                          valmax(k,1,1) = xcarmx(k,1)
                       endif
c
c			recherche des valeurs minimales
c
                       xcarmin(k,1) = dmin1(xcarlo(k),valmin(k,1))
                       if (xcarmin(k,1).lt.valmin(k,1)) then
                          nummin(k,1) = isimul
                          valmin(k,1) = xcarmin(k,1)
                       endif
                    endif
c
                end do
             endif
          else
             iechec = iechec + 1
c
             do  k = 2,nbsauv
c
c			calcul moyenne arithmetique
c
                 if (k.ne.33) then
                    xtotal(k,2) = xtotal(k,2) + xcarlo(k)
                    xmoyen(k,2) = xtotal(k,2)/dble(iechec)
c
c			calcul ecart-type arithmetique
c
                    xtotcr(k,2) = xtotcr(k,2) + xcarlo(k)**2
                    xtotmo(k,2) = xtotcr(k,2)/dble(iechec)
                    xectyp(k,2) = dsqrt(dabs(xtotmo(k,2) -
     +                                       xmoyen(k,2)**2))
c
c			recherche des valeurs maximales
c
                    xcarmx(k,2) = dmax1(xcarlo(k),valmax(k,1,2))
                    if (xcarmx(k,2).gt.valmax(k,1,2)) then
                       nummax(k,1,2) = isimul
                       valmax(k,1,2) = xcarmx(k,2)
                    endif
c
c			recherche des valeurs minimales
c
                    xcarmin(k,2) = dmin1(xcarlo(k),valmin(k,2))
                    if (xcarmin(k,2).lt.valmin(k,2)) then
                       nummin(k,2) = isimul
                       valmin(k,2) = xcarmin(k,2)
                    endif
                 endif
c
             end do
          endif
c
      end do
c
c		recherche des seconds cas pires
c
      rewind(unit= 310)

      do  isimul = 1,nbsimu
c
c		relecture du fichier de resultats
c
          read(310,1000) numero,(xcarlo(k), k = 2,nbsauv)
c
c		recherche de la valeur max.
c
          if (xcarlo(33).ge.dble(3)) then
             do  k = 2,nbsauv
                 if (k.ne.33) then
                    xcarmx(k,1) = dmax1(xcarlo(k),valmax(k,2,1))
c
c		validation si 2nde valeur max.
c
                    if ((xcarmx(k,1).gt.valmax(k,2,1)).and.
     +                  (isimul.ne.nummax(k,1,1))) then
                       nummax(k,2,1) = isimul
                       valmax(k,2,1) = xcarmx(k,1)
                    endif
                 endif
             end do
          endif
          if (xcarlo(33).lt.dble(3)) then
             do  k = 2,nbsauv
                 if (k.ne.32) then
                    xcarmx(k,2) = dmax1(xcarlo(k),valmax(k,2,2))
c
c		validation si 2nde valeur max.
c
                    if ((xcarmx(k,2).gt.valmax(k,2,2)).and.
     +                  (isimul.ne.nummax(k,1,2))) then
                       nummax(k,2,2) = isimul
                       valmax(k,2,2) = xcarmx(k,2)
                    endif
                 endif
             end do
          endif
c
      end do
c
c		edition ecran des resultats
c
      isorti = ielipt + ihyper
      write(6,*)
      write(6,*)
      write(6,2000)
      write(6,*)
      write(500,2010) nbsimu
      write(500,2020) isorti,ielipt,ihyper
      write(500,2042) iorbit(1),iorbit(2)
      write(500,2035) irebon
      write(500,2030) iechec
      write(500,*)
      write(6,2010) nbsimu
      write(6,2020) isorti,ielipt,ihyper
      write(6,2042) iorbit(1),iorbit(2)
      write(6,2035) irebon
      write(6,2030) iechec
      write(6,*)
      isucces = max0(nbsimu - iechec,1)
      write(6,3010) numsuc(1),100.d0*numsuc(1)/dble(isucces)
      write(6,3011) numsuc(2),100.d0*numsuc(2)/dble(isucces)
      write(6,3012) numsuc(3),100.d0*numsuc(3)/dble(isucces)
      write(6,3013) numsuc(4),100.d0*numsuc(4)/dble(isucces)
      write(6,3014) numsuc(5),100.d0*numsuc(5)/dble(isucces)
      write(500,*)
      write(500,3010) numsuc(1),100.d0*numsuc(1)/dble(isucces)
      write(500,3011) numsuc(2),100.d0*numsuc(2)/dble(isucces)
      write(500,3012) numsuc(3),100.d0*numsuc(3)/dble(isucces)
      write(500,3013) numsuc(4),100.d0*numsuc(4)/dble(isucces)
      write(500,3014) numsuc(5),100.d0*numsuc(5)/dble(isucces)      
      write(6,*)
      write(6,*)
      write(500,*)
      write(500,*)
      write(6,2100)
      write(500,2110)
      write(500,*)
      write(6,*)
      do  k = 2,nbsauv
          if (k.ne.33) then
             write(6,2200) variab(k),unites(k),
     +                               xmoyen(k,1),xectyp(k,1),
     +                               valmax(k,1,1),nummax(k,1,1),
     +                               valmin(k,1),nummin(k,1)
             write(500,2210) variab(k),unites(k),
     +                                 xmoyen(k,1),xectyp(k,1),
     +                                 valmax(k,1,1),nummax(k,1,1),
     +                                 valmax(k,2,1),nummax(k,2,1),
     +                                 valmin(k,1),nummin(k,1)
          endif
      end do
      write(6,*)
      write(6,*)
      if (iechec.ne.0) then
         write(6,2050)
         write(500,*)
         write(500,*)
         write(500,*)
         write(500,2050)
         write(500,*)
         write(500,*)
         do  k = 2,nbsauv
             if (k.ne.32) then
                write(6,2200) variab(k),unites(k),
     +                                  xmoyen(k,2),xectyp(k,2),
     +                                  valmax(k,1,2),nummax(k,1,2),
     +                                  valmin(k,2),nummin(k,2)
                write(500,2210) variab(k),unites(k),
     +                                    xmoyen(k,2),xectyp(k,2),
     +                                    valmax(k,1,2),nummax(k,1,2),
     +                                    valmax(k,2,2),nummax(k,2,2),
     +                                    valmin(k,2),nummin(k,2)
             endif
         end do
      endif
      write(6,*)
c
c		fermeture des fichiers
c
      close(unit= 310)
      close(unit= 310)
      close(unit= 500)
c
 1000 format(1x,i5,52(1x,d15.7))
c
 2000 format(1x,'Statistical Analysis')
 2010 format(1x,'runs number                          ',i5)
 2020 format(1x,'atmospheric exit number              ',i5,5x,
     +          'elliptic orbits       ',i5,2x,
     +          'hyperbolic orbits     ',i5)
 2042 format(1x,'unviable orbits number               ',i5,5x,
     +          'true anomaly < 180 ',i4)
 2030 format(1x,'failure cases (crash, capture)       ',i5)
 2035 format(1x,'number of skippings                  ',i5)
 2050 format(1x,'statistical results (OK on mission cases)')
 2100 format(1x,'parameter',27x,'mean  ',5x,'sigma',6x,'  max     num',
     +       2x,'    min     num')
 2110 format(1x,'parameter',27x,'mean  ',5x,'sigma',6x,'  max 1   num',
     +       2x,'  max 2   num',4x,'  min     num')
 2200 format(1x,a23,1x,a5,1x,f11.3,2x,f9.3,2x,f11.3,1x,i4,2x,f11.3,
     +       1x,i4,1x)
 2210 format(1x,a23,1x,a5,1x,2(f11.3,2x),3(f11.3,1x,i4,1x))
c
 3010 format(1x,'inclination       OK sur ',i5,' cases i.e. ',f8.3,
     +                                       ' % exit cases') 
 3011 format(1x,'insertion cost    OK sur ',i5,' cases i.e. ',f8.3,
     +                                       ' % exit cases') 
 3012 format(1x,'apoasis           OK sur ',i5,' cases i.e. ',f8.3,
     +                                       ' % exit cases')
 3013 format(1x,'periapsis         OK sur ',i5,' cases i.e. ',f8.3,
     +                                        ' % exit cases') 
 3014 format(1x,'mission Za+Zp+i+V OK sur ',i5,' cases i.e. ',f8.3,
     +                                       ' % exit cases')
c
      return
      end
