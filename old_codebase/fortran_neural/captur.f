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
      integer  icarlo,nbsimu,atmvar
      
      double precision ampli,wavlen
      
      common / varhor / atmvar,ampli,wavlen
      
      atmvar = 0
      wavlen = 500.
      ampli = 0.0
      
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
      
      close(847)
c
 4000 format(5(1x,d20.10))
 4004 format(101(1x,d20.10))
 8400 format(2(1x,d20.10))
 8402 format(7(1x,d20.10))
 
      stop
      end
