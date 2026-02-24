c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : loteri.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module realise le tirage des dispersions initiales. Les valeurs
c3    ainsi generees (hypothese de buit blanc gaussien ou uniforme) sont
c3    sauvegardees dans un fichier formatte a acces sequentiel.
c3
c3......................................................................
c4    variables d'entree
c4
c4    xgaela            R8    generateur aleatoire entre 0 et 1
c4    nbsimu            I4    nombre de simulations
c4......................................................................
c8    composants appelants
c8
c8    cisimu            INT   conditions generales de simulation
c8......................................................................
c9    composants appeles
c9
c9    bgauss            INT   generation d'un bruit blanc gaussien
c9    bunifo            INT   generation d'un bruit blanc uniforme
c9......................................................................
c10   commons utilises
c10
c10   disaer                  disperions atmopsheriques-aerodymamiques
c10   disini                  caracteristiques disperisions
c10   disnav                  erreurs de navigation
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  loteri  (xgalea,nbsimu)
c
      implicit none
c
      integer  nbsimu,
     +         i,atmvar,atmver
c
      double precision  dalfae,daltzd,daltit,dincid,dlongi,dlonzd,
     +                  dlatit,dlatzd,dnavad,dnaval,dnavaz,dnavla,
     +                  dnavlo,dnavpe,dnavvi,dnalti,dnazim,dndrag,
     +                  dnlati,dnlong,dnpent,dnvite,dvites,dvitzd,
     +                  dazimu,dazmzd,dpente,dpenzd,ddensi,droatm,
     +                  dcxeng,dxdrag,dczeng,dxlift,vagaus,vaunif,
     +                  xaleat,xgalea,dmasse,dmvehi,atmdis,wavlen,
     +                  ampli
c
      common / disaer / dincid,droatm,dxdrag,dxlift
      common / disini / daltzd,dlonzd,dlatzd,dvitzd,dazmzd,dpenzd
      common / dismas / dmasse
      common / disnav / dnaval,dnavlo,dnavla,dnavvi,dnavpe,dnavaz,
     +                  dnavad
     
      common / varhor / atmvar,ampli,wavlen
      common / varver / atmver,atmdis
c
      do  i = 1,nbsimu
c
          xaleat = xgalea
c
c		dispersions position
c
          call  bgauss (0.d0,1.d0,
     +                  xgalea,
     +                  vagaus)
          daltit = vagaus*daltzd

          call  bgauss (0.d0,1.d0,
     +                  xgalea,
     +                  vagaus)
          dlatit = vagaus*dlatzd

          call  bgauss (0.d0,1.d0,
     +                  xgalea,
     +                  vagaus)
          dlongi = vagaus*dlonzd
c
c		dispersions vitesse
c
          call  bgauss (0.d0,1.d0,
     +                  xgalea,
     +                  vagaus)
          dvites = vagaus*dvitzd

          call  bgauss (0.d0,1.d0,
     +                  xgalea,
     +                  vagaus)
          dazimu = vagaus*dazmzd

          call  bgauss (0.d0,1.d0,
     +                  xgalea,
     +                  vagaus)
          dpente = vagaus*dpenzd
c
c		dispersions atmosphere
c
          call  bunifo (0.d0,1.d0,
     +                  xgalea,
     +                  vaunif)
          ddensi = vaunif*droatm

          if (atmver.eq.1) then
          	ddensi = atmdis/100.
          endif
c
c		dispersions Cx - Cz
c
          call  bunifo (0.d0,1.d0,
     +                  xgalea,
     +                  vaunif)
          dcxeng = vaunif*dxdrag
          call  bunifo (0.d0,1.d0,
     +                  xgalea,
     +                  vaunif)
          dczeng = vaunif*dxlift
c
c		dispersions navigation position
c
          call  bgauss (0.d0,1.d0,
     +                  xgalea,
     +                  vagaus)
          dnalti = vagaus*dnaval

          call  bgauss (0.d0,1.d0,
     +                  xgalea,
     +                  vagaus)
          dnlati = vagaus*dnavla

          call  bgauss (0.d0,1.d0,
     +                  xgalea,
     +                  vagaus)
          dnlong = vagaus*dnavlo
c
c		dispersions navigation vitesse
c
          call  bgauss (0.d0,1.d0,
     +                  xgalea,
     +                  vagaus)
          dnvite = vagaus*dnavvi

          call  bgauss (0.d0,1.d0,
     +                  xgalea,
     +                  vagaus)
          dnazim = vagaus*dnavaz

          call  bgauss (0.d0,1.d0,
     +                  xgalea,
     +                  vagaus)
          dnpent = vagaus*dnavpe
c
c		erreur de mesure acceleration de trainee
c
          call  bgauss (0.d0,1.d0,
     +                  xgalea,
     +                  vagaus)
          dndrag = vagaus*dnavad
c
c		dispersion sur l'incidence
c
          call  bunifo (0.d0,1.d0,
     +                  xgalea,
     +                  vaunif)
          dalfae = vaunif*dincid
c
c		dispersion sur la masse (hypothese de dispersion uniforme)
c
          call  bunifo (0.d0,1.d0,
     +                  xgalea,
     +                  vaunif)
          dmvehi = vaunif*dmasse
c
c		sauvegarde des dispersions initiales et vol
c
          write(108,1000) i,xaleat,
     +                      daltit,dlongi,dlatit,
     +                      dvites,dazimu,dpente,
     +                      ddensi,
     +                      dcxeng,dczeng,
     +                      dnalti,dnlati,dnlong,
     +                      dnvite,dnazim,dnpent,
     +                      dndrag,dalfae,dmvehi
c
      end do
c
      rewind(unit= 108)
c
 1000 format(i5,1x,d15.7,18(1x,d15.7))
c
      return
      end
