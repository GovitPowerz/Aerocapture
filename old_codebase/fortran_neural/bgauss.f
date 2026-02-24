c1
c1    copyright (c) AEROSPATIALE 1993
c1......................................................................
c2    nom    : bgauss.f
c2    date   : 18/08/93
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module genere une variable aleatoire gaussienne de moyenne et
c3    d'ecart-type donnes a l'aide d'une suite arithmetique et a partir
c3    d'un generateur aleatoire compris entre 0 et 1 (obligatoirement en
c3    double precision).
c3
c3    nota : Cette methode possede un facteur de repetition de l'ordre
c3           de 300000.
c3......................................................................
c4    variables d'entree
c4
c4    esperx            R8    moyenne
c4    sygmax            R8    ecart-type
c4......................................................................
c5    variables de sortie
c5
c5    gnalea            R8    generateur aleatoire compris entre 0 et 1
c5......................................................................
c6    variables de sortie
c6
c6    vagaus            R8    variable aleatoire gaussienne comprise en
c6                            tre [-3.,+3.] a 3.sygma
c6......................................................................
c8    composants appelants
c8
c8    loteri           INT   tirage des dispersions et meconnaissances
c8......................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  bgauss (esperx,sygmax,
     +                    gnalea,
     +                    vagaus)
c
      implicit none
c
      integer  i
c
      double precision  a,esperx,fract,sygmax,vagaus,gnalea
c
      intrinsic  dint
c
      a = 0.d0
c
      do  i = 1,12
          fract = 9821.d0*gnalea + 0.211327d0
          gnalea = fract - dint(fract)
          a     = a + gnalea
      end do
c
      vagaus = (a - 6.d0)*sygmax + esperx
c
      return
      end
