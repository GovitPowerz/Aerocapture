c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : photra.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module permet de realiser des cliches de la trajectoire d'aero
c3    capture.
c3
c3......................................................................
c4    variables d'entree
c4
c4    positr(3)         R8    position absolue geocentrique spherique
c4    vitesr(3)         R8    vitesse relative locale spherique
c4    alfpil            R8    incidence commandee
c4    gitpil            R8    gite commandee
c4    temsim            R8    temps courant
c4    irebon            I4    indicateur de rebond atmospherique
c4......................................................................
c7    variables internes
c7
c7    iphase            I4    indicateur de phase de vol
c7                            iphase = 1 : phase de capture
c7                            iphase = 2 : phase de rebond
c7                            iphase = 3 : phase de sortie
c7    xlatit            R8    latitude
c7    xrayon            R8    rayon planete
c7......................................................................
c8    composants appelants
c8
c8    simmsr            INT   simulatiopn d'aerocapture
c8......................................................................
c9    composants appeles
c9
c9    frayon            INT   determination rayon planete
c9    orbito            INT   determination des parametres orbitaux
c9......................................................................
c10   commons utilises
c10
c10   trigon                  constantes trigonometriques
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  photra (positr,vitesr,alfpil,gitpil,somgit,
     +                    temsim,irebon,pdynan,romver,isimul,
     +                    itera)
c
      implicit none
c
      integer  irebon,isimul,itera,
     +         i,iphase
c
      double precision  positr(3),vitesr(3),alfpil,gitpil,somgit,
     +                  temsim,enerjr,vitrad,vittot,pdynan,
     +                  altitr,degrad,pi,xlatit,xorbit(13),
     +                  xphoto(24),
     +                  xrayon,romver
c
      common / trigon / degrad,pi
c
      intrinsic  dsin
c
      itera = itera + 1
c
c		determination altitude courante
c
      call  frayon (positr,
     +              altitr,xlatit)
c
c		determination des parametres orbitaux
c
      call  orbito (positr,vitesr,
     +              xorbit)
      call  energi (positr,vitesr,
     +              enerjr,vitrad,vittot)
     
c		test de la phase de vol
c
      if (irebon.eq.0) then
         if (altitr.gt.80.d3) then
            iphase = 1
         else
            iphase = 2
         endif
      else
         if ((positr(1) - xrayon).gt.80.d3) then
            iphase = 3
         else
            iphase = 2
         endif
      endif
c
c		sauvegarde des resultats
c
      xphoto(1) = temsim
c
      xphoto(2) = altitr/1.d3
      xphoto(3) = positr(2)/degrad
      xphoto(4) = xlatit/degrad
c
      xphoto(5) = vitesr(1)
      xphoto(6) = vitesr(2)/degrad
      xphoto(7) = vitesr(3)/degrad
c
      xphoto(8)  = xorbit(1)/1.d3
      xphoto(9)  = xorbit(2)
      xphoto(10) = xorbit(3)/degrad
      xphoto(11) = xorbit(4)/degrad
      xphoto(12) = xorbit(6)/1.d3
      xphoto(13) = xorbit(7)/1.d3
      xphoto(14) = iphase
      xphoto(15) = gitpil/degrad
      xphoto(16) = vitesr(1)*dsin(vitesr(2))
      xphoto(17) = alfpil/degrad
      xphoto(18) = somgit/degrad
      xphoto(19) = enerjr
      xphoto(20) = pdynan
      xphoto(21) = vitrad
      xphoto(22) = 0.5d0*romver*vitesr(1)**2/1.d3
      xphoto(23) = dble(isimul)
      xphoto(24) = (0.d0)
c
      write(400,1000) (xphoto(i), i= 1,24)
c
 1000 format(24(1x,d12.5))
c
      return
      end
