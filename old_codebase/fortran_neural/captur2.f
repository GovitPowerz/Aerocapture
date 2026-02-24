c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : captur.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module constitue le programme principal de l'outil de simulati-
c3    de la phase d'aerocapture de l'orbiter devant passer d'une orbite
c3    d'arrivee hyperbolique a une orbite de parking elliptique par sim-
c3    ple passage dans l'atmospehere de la planete cible.
c3    Cet outil de simulation traite essentiellement du guidage durant
c3    la phase d'aerocapture.
c3
c3......................................................................
c7    variables internes
c7
c7    icarlo            I4   indicateur de fonctionnement en Monte-Carlo
c7    nbsimu            I4   nombre de simulations a jouer
c7......................................................................
c9    composants appeles
c9
c9    cisimu            INT  conditions generalres de simulation
c9    simmsr            INT  simulation de l'aerocapture mission MSR
c9    statis            INT  traitement statistique des resultats
c9......................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      program  captur
c
      implicit none
c
      integer  icarlo,nbsimu,atmvar,atmver,j,i,posit1,nbtabg
      
      double precision ampli,wavlen,interv,atmdis,inter2,
     +                 tabepd(2,500,500),nbint2,interv,inter2,
     +                 enrmin,nbint3,gitatm(1000,2),
     +                 trajrf(1000,2000,7),giteit,atmosp
      
      character  *72 sufdin,sufdout
      
      common / tabgit / tabepd
      common / csttab / nbint2,interv,enrmin,nbint3,inter2
      common / varhor / atmvar,ampli,wavlen
      common / varver / atmver,atmdis
      common / tables / gitatm,trajrf,nbtabg
      
      read(5,*) sufdin
      read(5,*) sufdout
      read(5,*) atmvar
      read(5,*) ampli
      read(5,*) interv
      read(5,*) atmver
      read(5,*) inter2
      
      close(5)
      
      open(unit=202,file='../sorties/table_gite_essai',
     + form='formatted')
     
      open(unit=204,file='../sorties/table_pente_essai',
     + form='formatted')
     
      open(unit=847,file='../sorties/var_gitref',form='formatted')

      read(202,4000) nbint2,interv,enrmin,nbint3,inter2
      
      do i= 0,10000
      	read(202,4004,end=222) (tabepd(1,i,j) , j = 0,nbint2)
      	read(204,4004,end=222) (tabepd(2,i,j) , j = 0,nbint2)
      end do
            
 222  close(202)
      close(204)
      
      open(unit=200,file='../sorties/tablegite',form='formatted')
      
      posit1 = 1
      
      do i = 1,1000
      	read(200,8400,end=224) giteit,atmosp
      	gitatm(posit1,1)=atmosp
      	gitatm(posit1,2)=giteit
     	
     	posit1 = posit1 + 1
     	
      end do
     
 224  nbtabg=posit1-1
 
      if (atmvar.eq.1) then
      
      open(unit=812,file='../sorties/varatmos_hori'//sufdout,
     +    form='formatted')
     
      wavlen = 1.
      atmdis = 0.
      
      do while (wavlen.le.6000)
      
      	open(unit=5,file='aerocap'//sufdin,form='formatted')
      	
c
c		conditions generales de simulation
c
      	call  cisimu (icarlo,nbsimu)
c
c		simulation de l'aerocapture
c
      	if (icarlo.le.1) then
      	   call simmsr (icarlo,nbsimu)
      	endif
c
c		traitement statistique en cas de Monte-Carlo
c
      	if (icarlo.ge.1) then
      	   call  statis (nbsimu)
      	endif
      
      	close(5)
      	
      	wavlen = wavlen*dexp(interv)
      
      end do
      
      close(812)
      
      endif
      
      if (atmver.eq.1) then
      
      open(unit=814,file='../sorties/varatmos_vert'//sufdout,
     +    form='formatted')
     
      atmvar = 0
      atmdis = -100
      
      do while (atmdis.le.100)
      
      	open(unit=5,file='aerocap'//sufdin,form='formatted')
      	
c
c		conditions generales de simulation
c
      	call  cisimu (icarlo,nbsimu)
c
c		simulation de l'aerocapture
c
      	if (icarlo.le.1) then
      	   call simmsr (icarlo,nbsimu)
      	endif
c
c		traitement statistique en cas de Monte-Carlo
c
      	if (icarlo.ge.1) then
       	  call  statis (nbsimu)
      	endif
      
      	close(5)
      
      	atmdis = atmdis + inter2
      
      end do
      
      close(814)
      endif

 4000 format(5(1x,d20.10))
 4004 format(101(1x,d20.10))
 8400 format(2(1x,d20.10))
       
      stop
      end
