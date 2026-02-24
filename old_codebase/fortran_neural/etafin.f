c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : etafin.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module edite a l'ecran les conditions de fin de simulation
c3
c3......................................................................
c4    variables d'entree
c4
c4    xorbit(7)         R8    parametres orbitaux
c4    positr(3)         R8    position absolue repere geocentrique
c4    vitesr(3)         R8    vitesse relative repere local
c4    altmax(3)         R8    altitudes de parametres max.
c4    datmax(3)         R8    instants de parametres max.
c4    deltav(3)         R8    couts de changement d'orbite
c4    dvopti(3)         R8    couts de changement d'orbite optimal
c4    fluter(2)         R8    flux thermique courant et max
c4    fcharg(2)         R8    facteur de charge courant et max
c4    pdynam(2)         R8    pression dynamqiue courante et max
c4    somflu            R8    integrale de flux
c4    tcaptr            R8    duree de la phase de capture
c4    temsim            R8    temps courant
c4    tpcnum(3)         R8    duree max d'integration trajectoire
c4    trebon            R8    date de rebond
c4    zrebon            R8    altitude de rebond
c4    iprepr(2)         I4    compteur de commutations en preprogramme
c4    ifinal            I4    indicateur d'arret de simulation
c4    irebon            I4    indicateur de rebond
c4    nbroll            I4    nombre de renverses de roulis
c4......................................................................
c7    variables internes
c7
c7    enrjtf            R8    energie finale atteinte
c7    vitztf            R8    vitesse radiale finale
c7......................................................................
c8    composants appelants
c8
c8    simmsr            INT   simualtion aerocapture
c8......................................................................
c9    composants appeles
c9
c9    frayon            INT   calcul rayon planete
c9    enrtot            INT   abscisse du profil de gite commandee
c9......................................................................
c10   commons utilises
c10
c10   fensim
c10   gravit                  accelerations de pesanteur
c10   missio                  caracteristiques mission visees
c10   modgui                  nature de la simulation du guidage
c10   nrjvis                  parametres energetiques vises
c10   orbvis                  caracteristiques orbite visee
c10   period                  cadences integration...
c10   planet                  caracteristiques planete
c10   trigon                  constantes trigonometriques
c10   vlimit                  seuil de comparaison
c10   xvrent                  etat initial nominal
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  etafin (xorbit,positr,vitesr,altmax,datmax,deltav,
     +                    dvopti,fluter,fcharg,pdynam,somflu,somgit,
     +                    temsim,tcaptr,trebon,zrebon,iprepr,ifinal,
     +                    irebon,isimul,nbroll,itera)
c
      implicit none
c
      integer  ifinal,iprepr(2),irebon,isimul,nbroll,itera,
     +         i,j,natsim,numsim,numvis,numsuc
c
      double precision  xorbit(13),positr(3),vitesr(3),altmax(3),
     +                  datmax(3),deltav(4),dvopti(4),fluter(2),
     +                  fcharg(2),pdynam(2),somflu,somgit,temsim,
     +                  trebon,zrebon,
     +                  altitr,degrad,demiax,enrjfn,enrjtf,epsiln,
     +                  excorb,gomega,g0terr,g0mars,pi,positz,requat,
     +                  rpolar,tcaptr,tguida,tinteg,tnavig,tpilot,
     +                  tpredi,vitesz,vitzfn,vitztf,xaltfn,xazmfn,
     +                  xincli,xlatfn,xlonfn,xomega,xpenfn,xprepr,
     +                  zapoge,xlatir,xvitfn,zperig,tactiv,tsecur,
     +                  xphoto(24),errinc,errvit,errzap,errzpe,
     +                  enrtot
     
      integer atmvar,atmver
      
      double precision ampli,wavlen,atmdis

c
      common / fensim / numsim,numvis
      common / gravit / g0terr,g0mars
      common / missio / xaltfn,xlonfn,xlatfn,xvitfn,xpenfn,xazmfn
      common / modgui / natsim
      common / nrjvis / enrjfn,vitzfn
      common / orbvis / zapoge,zperig,demiax,excorb,xincli,gomega
      common / period / tnavig,tguida,tpilot,tpredi,tinteg
      common / planet / xomega(3),requat,rpolar
      common / succes / errinc,errvit,errzap,errzpe
      common / trigon / degrad,pi
      common / vlimit / epsiln
      common / xvrent / positz(3),vitesz(3)
      
      common / varhor / atmvar,ampli,wavlen
      common / varver / atmver,atmdis
c
      intrinsic  dble,dsin
c
      external  enrtot
c
      if (natsim.eq.2) then
         tcaptr = temsim + 1.d-7
      endif
      if (natsim.eq.3) then
         if (tcaptr.le.epsiln) tcaptr = 1.d-7
      endif
c
c		calculs preliminaires
c
       call  frayon (positr,
     +               altitr,xlatir)
c
      enrjtf = enrtot (positr,vitesr)
      vitztf = vitesr(1)*dsin(vitesr(2))
c
      tactiv = temsim - tguida*dble(iprepr(2))
      tsecur = tguida*dble(iprepr(1))
c
c		securisation des valeurs
c      
      if (tsecur.ge.temsim) then
         tsecur = temsim
      endif
      if (tactiv.le.0.d0) then
         tactiv = 0.d0
      endif
c
      xprepr = tguida*dble(iprepr(1) + iprepr(2))/temsim
c
c		edition des resultats
c
      if (atmvar.eq.1) then
      	write(812,4000) wavlen,deltav(4),enrjtf/1.d6
      endif
      
      if ((atmvar.eq.0).and.(atmver.eq.1)) then
      	write(814,4000) atmdis,deltav(4),enrjtf/1.d6
      endif
      
      write(6,*)
      write(6,1000) isimul
      write(6,*)
c
      write(6,1100) temsim
      write(6,1110) 100.d0*tguida*dble(iprepr(1))/temsim
      write(6,1111) 100.d0*tguida*dble(iprepr(2))/temsim
      if (iprepr(1).lt.epsiln) then
         write(6,1112) 0.d0
      else
         write(6,1112) 100.d0*tsecur/tactiv
      endif
      write(6,*)
      if (ifinal.eq.1) then
         write(6,3000)
c
c		sauvegarde de l'�nergie et du succ�s de la mission
c		pour la recherche de corridor
c		
	open(unit=260,file='../sorties/energie.finale',form='formatted')
	write(260,6000) enrjtf/1.d6,0,enrjfn/1.d6,(xorbit(7) - zapoge)/1.d3
	close(unit=260)
        write(6,*)
      else
c
c		sauvegarde de l'�nergie et du succ�s de la mission
c		pour la recherche de corridor
c
	open(unit=260,file='../sorties/energie.finale',form='formatted')
	write(260,6000) enrjtf/1.d6,1,enrjfn/1.d6,(xorbit(7) - zapoge)/1.d3
	close(unit=260)
      endif
      if (irebon.eq.1) then
          write(6,1600) zrebon/1.d3
          write(6,1610) trebon
          write(6,1620) nbroll
      endif
      write(6,1200) fluter(2)/1.d3,fcharg(2)/g0terr,pdynam(2)/1.d3
      write(6,1202) altmax(1)/1.d3,altmax(2)/1.d3,altmax(3)/1.d3
      write(6,1204) datmax(1),datmax(2),datmax(3)
      write(6,1210) somflu/1.d6
      write(6,1220) somgit/degrad
      write(6,*)
      write(6,1300) enrjtf/1.d6,vitztf
      write(6,*)
      write(6,1400) altitr/1.d3,vitesr(1)
      write(6,1410) positr(2)/degrad,vitesr(2)/degrad
      write(6,1420) xlatir/degrad, (-vitesr(3) + 2.d0*pi)/degrad
      write(6,*)
      write(6,1500) xorbit(1)/1.d3,(xorbit(1) - demiax)/1.d3
      write(6,1510) xorbit(2),xorbit(2) - excorb
      write(6,1520) xorbit(3)/degrad,(xorbit(3) - xincli)/degrad
      write(6,1530) xorbit(4)/degrad,(xorbit(4) - gomega)/degrad
      write(6,1535) xorbit(5)/degrad
      write(6,1536) xorbit(8)/degrad
      write(6,1540) xorbit(7)/1.d3,(xorbit(7) - zapoge)/1.d3
      write(6,1550) xorbit(6)/1.d3,(xorbit(6) - zperig)/1.d3
      write(6,1551) xorbit(9)
      write(6,1552) xorbit(10)/degrad
      write(6,1553) xorbit(11)
      write(6,1554) xorbit(12)
      write(6,1555) xorbit(13)
      
      write(6,*)
      if ((xorbit(6)).lt.0.d0) then
         write(6,3100)
         write(6,*)
      endif
      write(6,1690) deltav(4),dvopti(4)
      i = 1
      write(6,1700) i,deltav(1),dvopti(1)
      write(6,1700) i+1,deltav(2),dvopti(2)
      write(6,1700) i+2,deltav(3),dvopti(3)
      write(6,*)
      write(6,2000)
      write(6,*)
c
c		sauvegarde des parametres mission
c
      if (isimul.eq.numvis) then
         call  frayon (positz,
     +                 altitr,xlatir)
         write(330,1800) 0.,xaltfn/1.d3,xlonfn/degrad,xlatfn/degrad,
     +                      xvitfn,xpenfn/degrad,xazmfn/degrad,
     +                      enrjfn/1.d6,vitzfn,
     +                      demiax/1.d3,excorb,xincli/degrad,
     +                      gomega/degrad,
     +                      zapoge/1.d3,zperig/1.d3,
     +                      dvopti(1),dvopti(2),dvopti(3),
     +                      altitr/1.d3,positz(2)/degrad,
     +                      xlatir/degrad,
     +                      vitesz(1),vitesz(2)/degrad,
     +                      vitesz(3)/degrad
      endif
c
c		modification du fichier resultat de photra
c
      numsuc = 0

      if (dabs(xorbit(7) - zapoge).le.(errzap*1.d3)) then
         numsuc = numsuc + 1
      endif         
      if (dabs(xorbit(6) - zperig).le.(errzpe*1.d3)) then
         numsuc = numsuc + 3
      endif  
      if (dabs(xorbit(3) - xincli).le.(errinc/degrad)) then
         numsuc = numsuc + 5
      endif
         
      do  i = 1,itera
          backspace (unit= 400)
      end do
         
      do  i = 1,itera
          read (400,1900) (xphoto(j), j= 1,24)
          xphoto(24) = dble(numsuc)
          write(444,1900) (xphoto(j), j= 1,24)
      end do
         
c
 1000 format(1x,'     Fin de simulation ',i4)
 1100 format(1x,'duree aerocapture         ',f11.3,' s')
 1110 format(1x,'securisation cosinus gite ',f11.3,' %')
 1111 format(1x,'inhibition guidage longi  ',f11.3,' %')
 1112 format(1x,'securisation guidage longi',f11.3,' %')
 1200 format(1x,'valeurs maximales  ',f11.3,' kW/m2   ',f11.3,' g   ',
     +                                f11.3,' kPa')
 1202 format(1x,'                Z  ',f11.3,' km      ',f11.3,' km  ',
     +                                f11.3,' km')
 1204 format(1x,'                T  ',f11.3,' s       ',f11.3,' s   ',
     +                                f11.3,' s')
 1210 format(1x,'integrale de flux  ',f11.3,' MW/m2')
 1220 format(1x,'gite consommee     ',f11.3,' deg')  
 1300 format(1x,'valeurs finales E  ',f11.3,' MJ/kg dh/dt ',f11.3,
     +                                                    ' m/s')
 1400 format(1x,'altitude   ',f11.3,' km      vitesse    ',f11.3,' m/s')
 1410 format(1x,'longitude  ',f11.3,' deg     pente      ',f11.3,' deg')
 1420 format(1x,'latitude   ',f11.3,' deg     azimut     ',f11.3,' deg')
 1500 format(1x,'demi grand axe ',f11.3,' km     ecart ',f11.3,' km')
 1510 format(1x,'excentricite   ',f11.3,'        ecart ',f11.3)
 1520 format(1x,'inclinaison    ',f11.3,' deg    ecart ',f11.3,' deg')
 1530 format(1x,'longitude W    ',f11.3,' deg    ecart ',f11.3,' deg')
 1535 format(1x,'arugment Zp    ',f11.3,' deg')
 1536 format(1x,'anomalie vraie ',f11.3,' deg')
 1540 format(1x,'Z apoastre     ',f11.3,' km     ecart ',f11.3,' km')
 1550 format(1x,'Z periastre    ',f11.3,' km     ecart ',f11.3,' km')
 1551 format(1x,'vitesse infinie',f11.3,' m/s ')
 1552 format(1x,'nuinfini       ',f11.3,' deg ')
 1553 format(1x,'Vinfx          ',f11.3,' m/s ')
 1554 format(1x,'Vinfy          ',f11.3,' m/s ')
 1555 format(1x,'Vinfz          ',f11.3,' m/s ')
 1600 format(1x,'rebond atmosphere  ',f11.3,' km')
 1610 format(1x,'                T  ',f11.3,' s')
 1620 format(1x,'renverse de roulis ',i5)
 1690 format(1x,'cout de changement d''orbite ',f8.1,' m/s',1x,
     +          '(opt: ',f8.1,' m/s)')
 1700 format(1x,'manoeuvre',i2,'                 ',f8.1,' m/s',1x,
     +          '(opt: ',f8.1,' m/s)')
c
 1800 format(24(1x,d12.5))
 1900 format(24(1x,d12.5))
c
 3000 format(1x,'arret sur crash Orbiter')
 3100 format(1x,'Orbite non viable')
c
 2000 format(1x,'----------------------------------------------------')
c
c
 4000 format(4(1x,d20.10))
 6000 format(1x,d20.10,1x,I1,2(1x,d20.10))
 
      return
      end
