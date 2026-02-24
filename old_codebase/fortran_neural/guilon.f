c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : guilon.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Mechin D.
c2......................................................................
c3    Ce module determine la consigne de gite du guidage longitudinal
c3
c3......................................................................
c4    variables d'entree
c4
c4    positn(3)         R8    postion absolue courante geocentrique
c4    vitesn(3)         R8    vitesse relative locale
c4    acceln(2)         R8    accelerations aerodynamiques estimees
c4    ibounc            I4    indicateur de rebond
c4    iphase            I4    indicateur de phase du guidage longi
c4......................................................................
c5    variables d'entree-sortie
c5
c5    vitref            R8    vitesse radiale de consigne
c5    iprepr(2)         I4    indicateur de securisation du guidage
c5......................................................................
c6    variables de sortie
c6
c6    gitlon            R8    consigne de gite guidage longitudinal
c6    ilongi            I4    indicatuer d'activation guidage sortie
c6......................................................................
c8    composants appelants
c8
c8    guidag           INT   guidage type predicteur-correcteur
c8......................................................................
c9    composants appeles
c9
c9    guicap           INT   guidage longi en phase de capture
c9    guiext           INT   guidage longi en phase de sortie
c9......................................................................
c10   commons utilises
c10
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  guilon (gitpre,roguid,roexit,alfcom,
     +			  positn,vitesn,acceln,coefan,iphase,
     +                    vitref,iprepr,
     +                    dzapog,gitlon,temsim)
c
      implicit none
c
      integer  i,iphase,iprepr(2),iguida(2),natsim,natgnn
c
      double precision  gitpre,positn(3),vitesn(3),acceln(2),coefan(2),
     +                  vitref,dzapog(2),gitlon,alfcom,altitu,latitu,
     +                  degrad,pi,roguid,roexit,temsim,input(11)

      common / modgui / natsim,natgnn
      common / trigon / degrad,pi
c
c		guidage en vol equilibre, phase de capture
c
      if (natgnn.eq.1) then
        call guicap (positn,vitesn,acceln,coefan,gitpre,roguid,
     +             alfcom,
     +             iprepr,vitref,
     +             gitlon,iguida,temsim)
	      call  frayon (positn,
     +              altitu,latitu)
	      input(1) = altitu/1.d3
	      input(2) = positn(2)
	      input(3) = positn(3)
	      input(4) = vitesn(1)/1.d3
	      input(5) = vitesn(2)
	      input(6) = vitesn(3)
	      do i = 1,2
	         input(i+6) = acceln(i)
	      enddo
	      input(9) = roguid*100.d0
	      input(10) = gitpre
          input(11) = temsim
          write(957,1000) input
      else
        call guidnn (positn,vitesn,acceln,coefan,gitpre,roguid,
     +             temsim,gitlon)
      endif
c
 1000 format(1x,200(1x,d20.10))
      return
      end
