c1
c1    copyright (c) AEROSPATIALE 1999
c1......................................................................
c2    nom    : simmsr.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module realise la simulation de la phase d'aerocapture.
c3
c3......................................................................
c4    variables d'entree
c4
c4    icarlo            I4    indicateur de fonctionnement en Monte-Carlo
c4    nbsimu            I4    nombre de simulations a realiser
c4......................................................................
c7    variables internes
c7
c7    acceln(2)         R8    accelerations aerodynamiques estimees
c7    accelr(2)         R8    accelerations aerodynamiques reelles
c7    altitr            R8    altitude courante reelle
c7    altmax(3)         R8    altitudes de parametres max. (flux,...)
c7    coefan(2)         R8    coefficients aerodynamiques estimes
c7    datgui            R8    temps courant sur une periode de guidage
c7    datmax(3)         R8    dates de parametres max. (flux,...)
c7    datnav            R8    temps courant sur une periode de navigation
c7    datpil            R8    temps courant sur une periode de pilotage
c7    deltav(3)         R8    cout pour rejoindre l'orbite de parking
c7    ecartn(4)         R8    ecart final predit sur les contraintes
c7    ecartr(4)         R8    ecart courant reel sur les contraintes
c7    fcharg(2)         R8    facteur de charge courant et max
c7    finess            R8    finesse equilibree
c7    fluter(2)         R8    flux thermique courant et max
c7    gitcom            R8    gite commandee courante
c7    gitpre            R8    gite commandee precedente
c7    icallg            I4    indicateur d'appel au guidage
c7    icalln            I4    indicateur d'appel a la navigation
c7    icallp            I4    indicateur d'appel au pilotage
c7    icallr            I4    indicateur d'appel a la realite simulee
c7    icalls            I4    indicateur d'appel a la sauvegarde
c7    idebut            I4    indicateur d'initialisation du sequentiel
c7    iphase            I4    indicateur de phase du guidage longi
c7    iprepr(2)         I4    compteur de securisation du guidage
c7    irebon            I4    indicateur de premier rebond
c7    isatur            I4    indicateur de saturation de la commande
c7    isecur            I4    indicateur de securisation du guidage
c7    isimul            I4    numero de simulation en cours
c7    nbroll            I4    nombre de renverses de roulis
c7    pdynam(2)         R8    pression dynamique courante et max
c7    positn(3)         R8    position absolue estimee
c7    positr(3)         R8    position absolue reelle
c7    romver            R8    masse volumique atmosphere
c7    somflu            R8    integrale de flux
c7    temsim            R8    temps courant sur la simulation
c7    trebon            R8    date de rebond atmopsherique
c7    vitesn(3)         R8    vitesse relative estimee
c7    vitesr(3)         R8    vitesse relative reelle
c7    vitcom            R8    vitesse de gite commandee avant saturation
c7    vitmac            R8    nombre de Mach reel
c7    xaleat            R8    generateur aleatoire courant
c7    zrebon            R8    altitude de premier rebond
c7......................................................................
c8    composants appelants
c8
c8    aerocap           INT   programme principal aerocapture
c8......................................................................
c9    composants appeles
c9
c9    carltf            INT   sauvegarde des conditions finales en cas
c9                            de Monte-Carlo
c9    carltz            INT   sauvegarde des conditions initiales en cas
c9                            de Monte-Carlo
c9    cisimu            INT   condiitons initiales de simulation
c9    ergols            INT   cout pour changement d'orbite
c9    etafin            INT   edition ecran des resultats de simulation
c9    finmsr            INT   logique d'arret de simulation
c9    guidag            INT   guidage
c9    naviga            INT   navigation
c9    photra            INT   cliches iso-cadences de la trajectoire
c9    pilote            INT   pilotage
c9    realit            INT   integration trajectoire reelle
c9    result            INT   sauvegarde des parametres courants
c9    sequen            INT   sequentiel
c9......................................................................
c10   commons utilises
c10
c10   fenvis                  simulations a visualiser
c10   modres                  mode de sauvegarde des resultats
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  simmsr (icarlo,nbsimu)
c
      implicit none
c
      include '../include/dimensions.incl'
c
      integer  icarlo,nbsimu,
     +         ibounc,icallg,icalln,icallp,icallr,icalls,icallf,icaptr,
     +         icrash,idebut,ifinal,iguida(2),ilongi,indext,indrol,
     +         iphase,iprepr(2),irebon,isatur,isauve,isecur,isimul,
     +         nbroll,numsim,numvis,natsim,itera,
     +         isorti,ilater,irefer,indrvr,ncarlo
c
      double precision  acceln(2),accelr(2),alfcom,alfpil,altitr,
     +                  altmax(3),coefan(2),coefro,datmax(3),datgui,
     +                  datnav,datpil,datpho,deltav(4),dvopti(4),
     +                  dzapog(2),ecartn(4),ecartr(4),energn,energr,
     +                  fcharg(2),finess,fluter(2),gitcom,gitpil,
     +                  gitpre,pdynam(2),pdynan,positn(3),positr(3),
     +                  roexit,roguid,romver,sgngit,somflu,somgit,
     +                  tcaptr,temsim,tpctra,trebon,vitcom,vitesn(3),
     +                  vitesr(3),vitgit,vitmac,vitpre,vitref,vitszr,
     +                  xorbit(13),zrebon,temrol,trevrs,gpilpr,
     +			tlater,dtroll,romnom,alfini,gitini,positz,
     +                  vitesz
c
      common / fensim / numsim,numvis
      common / modgui / natsim
      common / modres / isauve
      common / traref / irefer
      common / modcrl / ncarlo
      common / oricom / alfini,gitini
      common / xvrent / positz(3),vitesz(3)

      if (nbsimu.gt.1) then
         ncarlo = 1
      else
         ncarlo = 0
      endif
c
      do  isimul = 1,nbsimu
c
c		initialisation des conditions de simulation
c
          call  inimsr (icarlo,isimul,
     +                  xorbit,ecartr,positr,vitesr,positn,vitesn,
     +                  altmax,datmax,fluter,fcharg,pdynam,alfpil,
     +                  coefro,gitpre,sgngit,somflu,somgit,temsim,
     +                  trebon,vitpre,vitref,zrebon,iprepr,ibounc,
     +                  icaptr,idebut,ifinal,iphase,irebon,isauve,
     +                  isecur,nbroll,indrol,indext,isorti,iguida,
     +			ilater,tlater,dtroll,itera,gpilpr,gitpil)
c
c		sauvegarde conditions initiales (Monte-Carlo)
c
          if (icarlo.eq.1) then
             call  carltz (positr,vitesr,isimul)
          endif
c
c		simulation aerocapture
c
          do while (ifinal.eq.0)
c
c		sequentiel
c
              call  sequen (temsim,datnav,datgui,datpil,datpho,
     +                      icalln,icallg,icallp,icallr,icalls,icallf,
     +                      idebut,indrvr,trevrs,temrol,iguida)
c
c		navigation
c
              if (icalln.eq.1) then
                 call  naviga  (positr,vitesr,alfpil,temsim,icarlo,
     +                          coefro,vitpre,ibounc,iphase,
     +                          ecartn,positn,vitesn,acceln,coefan,
     +                          energn,pdynan,roexit,roguid,tcaptr,
     +                          vitref,icrash,indext)
              endif
c
c		guidage
c
              if (icallg.eq.1) then

                 call  guidag (positn,vitesn,acceln,coefan,pdynan,
     +                         roexit,roguid,temsim,ibounc,iphase,
     +                         gitpil,
     +                         gitpre,sgngit,somgit,vitref,iprepr,
     +                         indrvr,nbroll,
     +                         alfcom,gitcom,gpilpr,trevrs,vitgit,
     +                         iguida,ilongi,indrol,isatur)

              endif
c
c		pilotage
c
              if (icallp.eq.1) then
                 call  pilote (positn,vitesn,alfcom,gitcom,gpilpr,
     +                         vitcom,datpil,
     +                         alfpil,gitpil,vitgit)
              endif
c
c		cliche de la trajectoire
c
              if (icallf.eq.1) then
                 call  photra (positr,vitesr,alfpil,gitpil,somgit,
     +                         temsim,irebon,pdynan,romver,isimul,
     +                         itera)
              endif

c
c		integration de trajectoire
c
              if (icallr.eq.1) then
c
                 call  realit (alfpil,gitpil,temsim,
     +                         positr,vitesr,altmax,datmax,fluter,
     +                         fcharg,pdynam,somflu,irebon,
     +                         xorbit,ecartr,accelr,altitr,energr,
     +                         finess,romver,romnom,trebon,vitmac,
     +                         vitszr,zrebon)
                 call  finmsr (altitr,temsim,vitszr,irebon,
     +                         ifinal)
              endif
c
c		sauvegarde des resultats
c
              if ((icalls.eq.1).and.(isimul.eq.numvis)) then
                 call  result (xorbit,ecartr,ecartn,positr,vitesr,
     +                         positn,vitesn,acceln,accelr,dzapog,
     +                         fluter,fcharg,pdynam,alfcom,alfpil,
     +                         coefro,energr,gitcom,gitpil,roexit,
     +                         roguid,romver,somflu,temsim,tpctra,
     +                         vitgit,vitmac,vitref,iguida,ilongi,
     +                         indext,indrol,isatur,isecur)
              endif
c
          end do
c
c		cout pour rallier l'orbite de parking
c
          call  ergols (xorbit,positr,vitesr,ifinal,
     +                  deltav,dvopti)
c
c		sauvegarde des resultats
c
          if (icarlo.eq.1) then
             call  carltf (xorbit,ecartr,positr,vitesr,altmax,datmax,
     +                     deltav,fluter,fcharg,pdynam,finess,somflu,
     +                     somgit,tcaptr,temsim,trebon,zrebon,iprepr,
     +                     ifinal,isimul,nbroll)
          endif
c
          if ((isimul.eq.numvis).and.(isauve.eq.1)) then
             call  result (xorbit,ecartr,ecartn,positr,vitesr,
     +                     positn,vitesn,acceln,accelr,dzapog,
     +                     fluter,fcharg,pdynam,alfcom,alfpil,
     +                     coefro,energr,gitcom,gitpil,roexit,
     +                     roguid,romver,somflu,temsim,tpctra,
     +                     vitgit,vitmac,vitref,iguida,ilongi,
     +                     indext,indrol,isatur,isecur)
c
             close (unit= 201)
             close (unit= 202)
             close (unit= 203)
             close (unit= 204)
             close (unit= 220)
c
          endif
c
         if (idebut.ne.1) then
            call  photra (positr,vitesr,alfpil,gitpil,somgit,
     +                    temsim,irebon,pdynan,romver,isimul,itera)
         endif
c
c		visualisation ecran des resultats de simulation
c
          call  etafin (xorbit,positr,vitesr,altmax,datmax,deltav,
     +                  dvopti,fluter,fcharg,pdynam,somflu,somgit,
     +                  temsim,tcaptr,trebon,zrebon,iprepr,ifinal,
     +                  irebon,isimul,nbroll,itera)
c
      end do
c
c		fermeture ou rembobinage des fichiers
c
      close (unit= 108)
      close (unit= 330)
      close (unit= 320)
      close (unit= 350)
      close (unit= 400)    
      close (unit= 444)
      if (icarlo.eq.1) then
         rewind (unit= 300)
         rewind (unit= 310)
      endif
      
      if (irefer.eq.1) then
         close (unit= 113)
      endif
c
      return
      end
