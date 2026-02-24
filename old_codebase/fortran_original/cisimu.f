c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : cisimu.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine les conditions generales de simulation
c3
c3......................................................................
c6    variables de sortie
c6
c6    icarlo            I4   indicateur de fonctionnement en Monte-Carlo
c6    nbsimu            I4   nombre de simulations a jouer
c6......................................................................
c7    variables internes
c7
c7    itirag            I4   indicateur de lecture ou tirage des disper-
c7                           sions
c7......................................................................
c8    composants appelants
c8
c8    captur            INT  programme de simulation d'aerocapture
c8......................................................................
c9    composants appeles
c9
c9    entree            INT  choix utilisateur de simulation
c9    lectci            INT  lecture des conditions de simulation
c9    loteri            INT  tirage des dispersions
c9    opnfic            INT  ouverture des fichiers donnees-resultats
c9......................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  cisimu (icarlo,nbsimu)
c
      implicit none
c
      integer  icarlo,nbsimu,
     +         iconfd,itirag

      double precision  xgalea
c
      iconfd = 0
c
      do  while (iconfd.eq.0)
c
c		choix de simulation par utilisateur
c
          call  entree (xgalea,icarlo,itirag,nbsimu)
c
c		ouverture des fichiers de donnees et de resultats
c
          call  opnfic (icarlo,
     +                  iconfd)
c
c		lecture des conditions de simulation
c
          if (iconfd.ne.0) then
             call  lectci  (iconfd)
          endif
c
      end do
c
c		tirage des dispersions aleatoires
c
      if (itirag.eq.1) then
         call  loteri  (xgalea,nbsimu)
      endif
c
      return
      end
