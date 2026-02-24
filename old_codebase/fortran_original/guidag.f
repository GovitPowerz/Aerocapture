c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : guidag.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine la consigne en gite, le schema de guidage rete
c3    nu etant de type prediceteur-correceteur (guidage par matrice de
c3    sensibilite) avec integration numerique de la trajectoire.
c3
c3......................................................................
c4    variables d'entree
c4
c4    positn(3)         R8    position absolue estimee
c4    vitesn(3)         R8    vitesse relative estimee
c4    acceln(2)         R8    accelerations aerodynamiques estimees
c4    coefan(2)         R8    coefficients aerodynamiques estimes
c4    pdynan            R8    pression dynamique estimee
c4    roexit            R8    densite atmospherique finale predite
c4    rogui             R8    densite atmospherique courante estimee
c4    temsim            R8    temps courant 
c4    ibounc            I4    indicateur de rebond  
c4    iphase            I4    indicateur de phase du guidage
c4......................................................................
c5    variables d'entree-sortie
c5
c5    gitpre            R8    gite commandee au cycle precedent
c5    sgngit            R8    signe de la gite commandee 
c5    somgit            R8    cumul des increments de gite commandee
c5    vitref            R8    vitesse radiale de consigne
c5    iprepr(2)         I4    compteur de securisation du guidage
c5    nbroll            I4    nombre de renverses de roulis
c5......................................................................
c6    variables de sortie
c6
c6    dzapog(2)         R8    ecart apoastre courant et guide
c6    alfcom            R8    incidence commandee sur profil
c6    gitcom            R8    gite commandee courante
c6    vitgit            R8    vitesse de gite comamndee avant saturation
c6    iguida(2)         I4    indicateurs de securisation du guidage
c6    ilongi            I4    indicateur de guidage effectif en capture
c6    indrol            I4    indicateur de renverse de roulis
c6    isatur            I4    indicateur de saturation en vitesse
c6......................................................................
c7    variables internes
c7
c7    ilongi            I4    indicateur d'activation du guidage
c7......................................................................
c8    composants appelants
c8
c8    simmsr           INT    simulation aerocapture
c8......................................................................
c9    composants appeles
c9
c9    guialf           INT    guidage en incidence
c9    guilat           INT    guidage lateral
c9    guilon           INT    guidgae longitudinal    
c9    vigite           INT    saturation de la commande
c9......................................................................
c10   commons utilises
c10
c10   congui                  seuils d'activation du guidage
c10   loglat                  seuil energetique pour guidage lateral
c10   modecr                  edition ecran intermediaires
c10   trigon                  constantes trigonometriques
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  guidag (positn,vitesn,acceln,coefan,pdynan,
     +                    roexit,roguid,temsim,ibounc,iphase,
     +                    gitpil,
     +                    gitpre,sgngit,somgit,vitref,iprepr,
     +                    indrvr,nbroll,
     +                    alfcom,gitcom,gpilpr,trevrs,vitgit,
     +                    iguida,ilongi,indrol,isatur)
c
      implicit none
c
      integer  ibounc,iphase,iprepr(2),nbroll,iguida(2),ilongi,indrol,
     +         isatur,ilater,irefer,natman,
     +         iecran,natsim,indrvr,icarlo,rolway
c
      double precision  positn(3),vitesn(3),acceln(2),coefan(2),pdynan,
     +                  roexit,roguid,temsim,gitpre,sgngit,somgit,
     +                  vitref,dzapog(2),alfcom,gitcom,vitgit,
     +                  degrad,enrjlt,enrlat,gitlon,pdacti,pdinib,pi,
     +                  gitref,srefer,vgitmx,xmasse,enrtot,excent,xj2,
     +                  xmug,sgnpre,hpp,acgrav,pdyneq,xorbit(13),
     +                  xinccr,vitrel,vitrad,trevrs,
     +                  tnavig,tguida,tpilot,tpredi,tinteg,
     +                  altitu,xlatit,gpilpr,gitpil
     
c
      common / capsul / srefer,vgitmx,xmasse
      common / congui / pdacti,pdinib
      common / geoide / excent,xj2,xmug
      common / loglat / enrlat(2)
      common / modecr / iecran
      common / modaga / natman
      common / modgui / natsim
      common / period / tnavig,tguida,tpilot,tpredi,tinteg
      common / trigon / degrad,pi
      common / traref / irefer
      common / gitrfr / gitref
      common / modcrl / icarlo
      common / rolchg / rolway
c
      external  enrtot
c
c		initialisations
c
      dzapog(1) = 1.d33
      dzapog(2) = 1.d33
     
      
      sgnpre = sgngit 
      gpilpr = gitpil
c
c		consigne de guidage en incidence
c
      call  guialf (positn,vitesn,roguid,
     +              alfcom)
c
c		tests d'inhibition du guidage longitudinal
c
      enrjlt = enrtot (positn,vitesn)  
         
      if (natman.eq.1) then
c
c		manoeuvre d'aerocapture
c
         if ((enrjlt.le.pdacti).and.(enrjlt.ge.pdinib)) then
            ilongi    = 1
         else
            ilongi    = 0
            iprepr(2) = iprepr(2) + 1
         endif
      else
c
c		manoeuvre d'aero-gravity assist
c    
         if (ibounc.eq.0) then
            if (pdynan.lt.pdacti) then
               ilongi = 0
            else
               ilongi = 1
            endif
         else
            if (pdynan.lt.pdinib) then
               ilongi = 0
            else
              ilongi = 1
            endif            
         endif
         if (ilongi.eq.0) then
            iprepr(2) = iprepr(2) + 1
         endif
      endif

      ilongi = ilongi*iguida(1)
c
c		sauvegarde de la trajectoire de reference
c
      if ((irefer.eq.1).and.(icarlo.eq.0)) then
      
         ilongi    = 0
         iguida(1) = 0
         iguida(2) = 0

         vitrel = vitesn(1)
         vitrad = vitesn(1)*dsin(vitesn(2))
         pdyneq = 0.5d0*roguid*vitrel**2
         acgrav = xmug/positn(1)**2
         
         enrjlt = enrtot (positn,vitesn)     
          
         hpp = dcos(gitref)*srefer*coefan(2)*pdyneq/xmasse -
     +        (acgrav - (vitrel**2/positn(1)))*dcos(vitesn(2))
      
         call  orbito (positn,vitesn,
     +                 xorbit)
         call  frayon (positn,
     +                 altitu,xlatit)      
         xinccr = xorbit(3)
      
         write(113,777) enrjlt/1.d6,pdyneq,vitrad,hpp,xinccr/degrad,
     +                 temsim,dcos(gitref) 
     
  777    format(7(1x,1pe23.16))
      endif
c
c		Logique de guidage longitudinal (seuils en Pdyn)
c
      if (irefer.eq.1) then
         gitcom = gitref
      else
         if (ilongi.eq.0) then
            gitlon = dabs(gitref)
         else
c
            call  guilon (gitpre,roguid,roexit,alfcom,
     +		          positn,vitesn,acceln,coefan,iphase,
     +                    vitref,iprepr,
     +                    dzapog,gitlon,temsim)
         endif
      endif
c
c		logique de guidage lateral (seuil en Energie ou Pdyn)
c
      if (natman.eq.1) then
         if ((enrjlt.le.enrlat(1)).and.(enrjlt.ge.enrlat(2))) then
            ilater = 1
         else
            ilater = 0
         endif
      else
         if (enrlat(1).lt.0.d0) then
c
c		activation du lateral apres rebond sur critere Pdyn
c
            if (ibounc.eq.1) then
               if ((pdynan.ge.enrlat(2)).and.
     +             (pdynan.le.-enrlat(1))) then
                  ilater = 1
               else
                  ilater = 0
               endif
            else
               ilater = 0
            endif        
         else
c
c		activation du lateral avant rebond sur critere Pdyn
c
            if (ibounc.eq.0) then
               if (pdynan.lt.enrlat(1)) then
                  ilater = 0
               else
                  ilater = 1
               endif
            else
               if (pdynan.lt.enrlat(2)) then
                  ilater = 0
               else
                  ilater = 1
               endif            
            endif         
         endif
      endif
      
      ilater = ilater*iguida(2)     
c      
      if (ilater.eq.1) then
	 call  guilat (positn,vitesn,gitlon,temsim,gitpre,
     +                 sgngit,nbroll,
     +                 trevrs,indrvr,indrol)
         if (indrvr.eq.1) then
            iguida(2) = 0
         endif
      else
         sgngit = sgnpre
      endif
c     
      if (irefer.eq.0) then
         if (iguida(1)*iguida(2).eq.1) then
            gitcom = gitlon*sgngit
         else
            if (indrvr.eq.1) then
               if (rolway.eq.1) then
               if (sgngit.gt.0.d0) then
                  gitcom = gitpre + vgitmx*tguida
                  write(6,*) gitcom/degrad,gitpre/degrad,sgngit,rolway
               else
                  gitcom = gitpre - vgitmx*tguida
                  write(6,*) gitcom/degrad,gitpre/degrad,sgngit,rolway
               endif
               else
               if (sgngit.gt.0.d0) then
                  gitcom = gitpre - vgitmx*tguida
                  if (gitcom.lt.-pi) then
                  	gitcom=gitcom+2*pi
                  endif
                  write(6,*) gitcom/degrad,gitpre/degrad,sgngit,rolway
               else
                  gitcom = gitpre + vgitmx*tguida
                  if (gitcom.gt.pi) then
                  	gitcom=gitcom-2*pi
                  endif
                  write(6,*) gitcom/degrad,gitpre/degrad,sgngit,rolway
               endif
               endif
            endif
         endif
      endif
c 
c		saturation de la commande sur criter de vitesse
c
      call  vigite (gitcom,gitpre,somgit,
     +              vitgit,isatur)
c
c      write(345,1000) temsim,1.d3*tcpu(1),1.d3*tcpu(2),
c     +                1.d3*(tcpu(2) - tcpu(1))
c     
c 1000 format(4(1x,d10.5))
 
      return
      end
      
      
      
