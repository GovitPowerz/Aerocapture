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
c6    vanuif            R8    variable aleatoire uniforme comprise entre
c6                            -1 et 1
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
      subroutine  bunifo (esperx,sygmax,
     +                    gnalea,
     +                    vaunif)
c
      implicit none
c
      double precision  esperx,fract,sygmax,vaunif,gnalea
c
      intrinsic  dint
c
      fract = 9821.d0*gnalea + 0.211327d0
      gnalea = fract - dint(fract)
c
      vaunif = 2.d0*gnalea - 1.d0
c
      return
      end
