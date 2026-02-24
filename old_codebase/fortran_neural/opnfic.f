c1
c1    copyright (c) AEROSPATIALE 1999
c1......................................................................
c2    nom    : opnfic.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module realise l'ouverture des fichiers de donnees et de resul-
c3    tats. A noter que les fichiers de donnees sont fermees apres leur
c3    lecture, les parametres lus (module lectci) etant place dans des
c3    commons.
c3
c3......................................................................
c4    variables d'entree
c4
c4    icarlo            I4    indicateur d'utilisation en Monte-Carlo
c4    numvis            I4    numero de simulation a visualiser
c4......................................................................
c8    composants appelants
c8
c8    cisimu            INT  conditions generales de simulation
c8......................................................................
c9    composants appeles
c9
c9    strlen            INT  longueur d'une chaine de caracteres
c9......................................................................
c10   commons utilises
c10
c10   fensim                 indications des simulations a sauvegarder
c10   ficdat                 suffixes des noms de fichiers de donnees
c10   ficres                 suffixes des noms de fichiers de resultats
c10   modres                 indicateur de sauvegarde des resultats
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  opnfic  (icarlo,
     +                     iconfd)
c
      implicit none
c
      integer  *4  icarlo,iconfd,
     +             isauve,lonfic,lenref,numsim,numvis,natsim,natpla,
     +             irefer,
     +             strlen
     
      double precision  xomega,requat,rpolar

      character  *1  sufnmr
      character  *2  sufnms
      character  *3  sufnmt
      character  *4  sufnmu
      character  *5  sufnum
      character  *72 sufaer,sufatm,sufdis,sufgui,sufinc,suflot,sufmis,
     +               sufmsr,sufnav,sufren,sufres,sufsuc,nomfic,sufgnn
c
      logical  existf
c
      common / fensim / numsim,numvis
      common / ficdat / sufaer,sufatm,sufdis,sufgui,sufinc,suflot,
     +                  sufmis,sufmsr,sufnav,sufren,sufsuc,sufgnn
      common / ficres / sufres
      common / modgui / natsim
      common / modres / isauve
      common / traref / irefer
      
      common / planet / xomega(3),requat,rpolar,natpla
c
      external  strlen
c
      lenref = strlen(sufres)
c
      iconfd = 1
c
c		test d'existence des ficheirs de donnees
c
      nomfic = '../donnees/capsule'//sufmsr
      lonfic = strlen(nomfic)
      inquire( file= nomfic(1:lonfic), exist= existf)
      if (.not.existf) then
         iconfd = 0
         write(6,1000) nomfic(1:lonfic)
      endif
      nomfic = '../donnees/mission'//sufmis
      lonfic = strlen(nomfic)
      inquire( file= nomfic(1:lonfic), exist= existf)
      if (.not.existf) then
         iconfd = 0
         write(6,1000) nomfic(1:lonfic)
      endif       
      nomfic = '../donnees/rentree'//sufren
      lonfic = strlen(nomfic)
      inquire( file= nomfic(1:lonfic), exist= existf)
      if (.not.existf) then
         iconfd = 0
         write(6,1000) nomfic(1:lonfic)
      endif
      nomfic = '../donnees/aerodynamique'//sufaer
      lonfic = strlen(nomfic)
      inquire( file= nomfic(1:lonfic), exist= existf)
      if (.not.existf) then
         iconfd = 0
         write(6,1000) nomfic(1:lonfic)
      endif 
      nomfic = '../donnees/atmosphere'//sufatm
      lonfic = strlen(nomfic)
      inquire( file= nomfic(1:lonfic), exist= existf)
      if (.not.existf) then
         iconfd = 0
         write(6,1000) nomfic(1:lonfic)
      endif
      nomfic = '../donnees/dispersions'//sufdis
      lonfic = strlen(nomfic)
      inquire( file= nomfic(1:lonfic), exist= existf)
      if (.not.existf) then
         iconfd = 0
         write(6,1000) nomfic(1:lonfic)
      endif
      nomfic = '../donnees/navigation'//sufnav
      lonfic = strlen(nomfic)
      inquire( file= nomfic(1:lonfic), exist= existf)
      if (.not.existf) then
         iconfd = 0
         write(6,1000) nomfic(1:lonfic)
      endif
      nomfic = '../donnees/guidage'//sufgui
      lonfic = strlen(nomfic)
      inquire( file= nomfic(1:lonfic), exist= existf)
      if (.not.existf) then
         iconfd = 0
         write(6,1000) nomfic(1:lonfic)
      endif
      nomfic = '../donnees/pilote'//sufmsr
      lonfic = strlen(nomfic)
      inquire( file= nomfic(1:lonfic), exist= existf)
      if (.not.existf) then
         iconfd = 0
         write(6,1000) nomfic(1:lonfic)
      endif
      if (irefer.eq.0) then
         nomfic = '../donnees/tables_energie_gains'//sufgui
         lonfic = strlen(nomfic)
         inquire( file= nomfic(1:lonfic), exist= existf)
         if (.not.existf) then
            iconfd = 0
            write(6,1000) nomfic(1:lonfic)
         endif
      endif
      nomfic = '../donnees/incidence'//sufinc
      lonfic = strlen(nomfic)
      inquire( file= nomfic(1:lonfic), exist= existf)
      if (.not.existf) then
         iconfd = 0
         write(6,1000) nomfic(1:lonfic)
      endif
      nomfic = '../donnees/succes'//sufsuc
      lonfic = strlen(nomfic)
      inquire( file= nomfic(1:lonfic), exist= existf)
      if (.not.existf) then
         iconfd = 0
         write(6,1000) nomfic(1:lonfic)
      endif
cc      nomfic = '../donnees/transition'//sufren
cc      lonfic = strlen(nomfic)
cc      inquire( file= nomfic(1:lonfic), exist= existf)
cc      if (.not.existf) then
cc         iconfd = 0
cc         write(6,1000) nomfic(1:lonfic)
cc      endif
c
      if (iconfd.eq.1) then
c
c		ouverture des fichiers de donnees
c
         open(unit= 100, file= '../donnees/capsule'//sufmsr,
     +                   form= 'formatted')
         open(unit= 101, file= '../donnees/mission'//sufmis,
     +                   form= 'formatted')
         open(unit= 102, file= '../donnees/rentree'//sufren,
     +                   form= 'formatted')
         open(unit= 104, file= '../donnees/aerodynamique'//sufaer,
     +                   form= 'formatted')
         open(unit= 105, file= '../donnees/atmosphere'//sufatm,
     +                   form= 'formatted')
         open(unit= 106, file= '../donnees/dispersions'//sufdis,
     +                   form= 'formatted')
         open(unit= 107, file= '../donnees/navigation'//sufnav,
     +                   form= 'formatted')
         open(unit= 108, file= '../donnees/loterie'//suflot,
     +                   form= 'formatted')
         open(unit= 109, file= '../donnees/guidage'//sufgui,
     +                   form= 'formatted')
         open(unit= 115, file= '../donnees/pilote'//sufmsr,
     +                   form= 'formatted')
         open(unit= 113, file= '../donnees/tables_energie_gains'
     +                                    //sufgui,
     +                   form= 'formatted')
         open(unit= 110, file= '../donnees/incidence'//sufinc,
     +                   form= 'formatted')
         open(unit= 111, file= '../donnees/succes'//sufsuc,
     +                   form= 'formatted')
         open (unit= 923, file= '../donnees/nn_param'//sufgnn,
     +                    form= 'formatted')
         open (unit= 924, file= '../donnees/nn_param2'//sufgnn,
     +                    form= 'formatted')
         if (natsim.eq.4) then
            open(unit= 112, file= '../donnees/commande_maxtom'//sufgui,
     +                      form= 'formatted')      
         endif
cc         open(unit= 114, file= '../donnees/transition'//sufren,
cc     +                   form= 'formatted')
c
c		ouverture des fichiers de resultats
c
         if (icarlo.eq.0) then
            if (isauve.eq.1) then
               open(unit= 201, file= '../sorties/cinematique'//sufres,
     +                         form= 'formatted')
               open(unit= 202, file= '../sorties/divers'//sufres,
     +                      form= 'formatted')
               open(unit= 203, file= '../sorties/guidage'//sufres,
     +                         form= 'formatted')
               open(unit= 204, file= '../sorties/commande'//sufres,
     +                         form= 'formatted')
               open(unit= 220, file= '../sorties/prediction'//sufres,
     +                         form= 'formatted')
            endif
         else
c
            if (isauve.eq.1) then
               if (numvis.le.9) then
                  write(sufnmr,'(i1)') numvis
                  sufnum = '_000'//sufnmr
               else
                  if (numvis.le.99) then
                     write(sufnms,'(i2)') numvis
                     sufnum = '_00'//sufnms
                  else
                     if (numvis.le.999) then
                        write(sufnmt,'(i3)') numvis
                        sufnum = '_0'//sufnmt
                     else
                        write(sufnmu,'(i4)') numvis
                        sufnum = '_'//sufnmu
                     endif
                  endif
               endif
c
               open(unit= 201, file= '../sorties/cinematique'//
     +                               sufres(1:lenref)//sufnum,
     +                         form= 'formatted')
               open(unit= 202, file= '../sorties/divers'//
     +                                sufres(1:lenref)//sufnum,
     +                         form= 'formatted')
               open(unit= 203, file= '../sorties/guidage'//
     +                               sufres(1:lenref)//sufnum,
     +                         form= 'formatted')
               open(unit= 204, file= '../sorties/commande'//
     +                               sufres(1:lenref)//sufnum,
     +                         form= 'formatted')
               open(unit= 220, file= '../sorties/prediction'//
     +                               sufres(1:lenref)//sufnum,
     +                         form= 'formatted')
            endif
	 endif 
	        
     	 open(unit= 400, file= '../sorties/photo_temp'//sufres,
     +                   form= 'formatted')
     	 open(unit= 444, file= '../sorties/photo'//sufres,
     +                   form= 'formatted')
         open(unit= 330, file= '../sorties/mission'//sufres,
     +                   form= 'formatted')     
         open(unit= 300, file= '../sorties/initial'//sufres,
     +                   form= 'formatted')
         open(unit= 310, file= '../sorties/final'//sufres,
     +                   form= 'formatted')
         open(unit= 500, file= '../listings/resume'//sufres,
     +                   form= 'formatted')
      endif
c
 1000 format(1x,'fichier non existant :',a72)
 2000 format(6(1x,d12.5))
c
      return
      end
