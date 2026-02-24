c1
c1    copyright (c) EADS Launch Vehicles 2005
c1......................................................................
c2    nom    : guidnn.f
c2    date   : 18/02/06
c2    IV     : 1
c2    IE     : 1
c2    auteur : G. Gelly
c2......................................................................
c3    Ce module elabore la consigne de gite a partir du schema EAGLE
c3......................................................................
c4    variables d'entree
c4
c4    xm(7)         R8    etat courant
c4    xlm           R8    force de portance
c4    xdm           R8    force de trainee
c4    xmac          R8    nombre de Mach
c4    alfcom        R8    incidence commandee
c4    datgnc        R8    temsp courant Pvol
c4    nphase        I4    phase du guidage longi
c4    giteci        R8    gite commandee precedente
c4    siggit        R8    signe de la gite
c4......................................................................
c6    variables de sortie
c6
c6    gitecn        R8
c6    gitesc        R8
c6......................................................................
c7    variables internes
c7......................................................................
c8    composants appelants
c8
c8    guihyp        INT   guidage hypersonique    
c8......................................................................
c9    composants appeles
c9
c9    calpln        INT   portee longitudinale
c9    iglgui        INT
c9    iglsol        INT
c9......................................................................
c10   commons utilises
c10
c10   etatf
c10   physi
c10   temgnc
c10   
c10   geagle
c10   peagle
c10.....................................................................
c13   norme logicielle GENE S320
c13
c13   non
c13.....................................................................
c
      subroutine  guidnn (positn,vitesn,acceln,coefan,gitpre,roguid,
     +                    temsim,gitlon,gitlon2)
c
      implicit none
c        
      include '../donnees/param_algo'
c
      integer  i,j
c        
      double precision  positn(3),vitesn(3),acceln(2),coefan(2),roguid,
     +                  gitlon,gitpre,degrad,pi,vitrad,integ(6),
     +                  xorbit(13),pdyneq,temsim,dtanh,count,gitlon2,
     +                  acgrav,rayvec,excent,xj2,xmug,compre,dabs,
     +                  zapoge,zperig,demiax,excorb,xincli,gomega
c
      common / orbvis / zapoge,zperig,demiax,excorb,xincli,gomega
      common / trigon / degrad,pi
      common / geoide / excent,xj2,xmug
      common / tempog / compre
      common / tmpint / count,integ
c       
      intrinsic  dtanh,dabs,dasinh
c
      rayvec = positn(1)
      vitrad = vitesn(1)*dsin(vitesn(2))
      acgrav = xmug/positn(1)**2
      pdyneq = 0.5d0*roguid*vitesn(1)**2

      call  orbito  (positn,vitesn,
     +                     xorbit)
c
      inputn1(1) = (xorbit(2)-1.d0)
      inputn1(2) = (xorbit(3)-xincli)/degrad*3.d0/5.d0
      inputn1(3) = 2.d0*(vitrad/1.d3+1.2d0)/1.5d0-1.d0
      inputn1(4) = (-xmug/(2*xorbit(1))/6.d6)
      inputn1(5) = (vitesn(1)/3.d3-1.5d0)*2.d0
      inputn1(6) = dsqrt(acceln(1)**2+acceln(2)**2)/20.d0-1.d0
c
      count = 1.d0
      do j = 1,n1hid1
         n1outhid1(j) = 0.d0
         do i = 1,n1input
            n1outhid1(j) = n1outhid1(j)+n1lw1(j,i)*inputn1(i)
         enddo
         n1outhid1(j) = dtanh(n1outhid1(j)+n1bias1(j))
         count = count+1.d0
      enddo
c
      do j = 1,n1output
         n1out(j) = 0.d0
         do i = 1,n1hid1
            n1out(j) = n1out(j)+n1lw4(j,i)*n1outhid1(i)
         enddo
         n1out(j) = dasinh(n1out(j)+n1bias4(j))
      enddo
c
c
c
      gitlon = datan2(n1out(1),n1out(2))
c      gitlon = 2.d0*pi*n1out(1)
c      gitlon2 = n2out(1)
      gitlon2 = 1.d0
c         gitlon = pi*n1out(1)
c
c      if (gitpre.ne.compre) then
c         if (compre.lt.0.d0) then
c            gitlon = -dabs(gitlon)
c         else
c            gitlon = dabs(gitlon)
c         endif
c      endif
c
      compre = gitlon
c
c      gitlon = datan2(output(1),output(2))
c      gitlon = 2*pi*dsin(output(1))
c      write(6,*) (input(i), i = 1,9)
c      write(695,1000) input(1),demiax-xorbit(1)
c      write(696,1000) output,gitlon*180.d0/pi
c      write(957,1002) (inputn1(i), i = 1,6)
c      write(696,1000) output,gitlon*180.d0/pi
c      write(693,*) temsim,gitlon*180.d0/pi,n1out(1),n2out(1)
c 1000 format(1x,i,1x,200(1x,d20.10))
 1002 format(1x,200(1x,d20.10))
c        
      return
      end
