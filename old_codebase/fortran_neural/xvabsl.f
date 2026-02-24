c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : xvabsl.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine les
c3
c3    NOTA  les vecteurs position et vitesse en entree sont fournis en
c3          coordonnees spheriques
c3
c3......................................................................
c4    variables d'entree
c4
c4    xposit(3)         R8    position absolue goecentrique
c4    xvites(3)         R8    vitesse relative locale
c4......................................................................
c6    variables de sortie
c6
c6    posita(3)         R8    position absolue geocentrqiue
c6    vitesa(3)         R8    vitesse absolue geocentrique
c6......................................................................
c7    variables internes
c7
c7    vitese(3)         R8    vitesse d'entrainement
c7    vitesl(3)         R8    vitesse relative locale cartesienne
c7    vitesr(3)         R8    vitesse relative geocentrique
c7
c7......................................................................
c8    composants appelants
c8
c8    orbito            INT   parametres orbitaux
c8......................................................................
c9    composants appeles
c9
c9    cartes            INT   passage coordonnees cartesiennes
c9    matvec            INT   produit matrice-vecteur
c9    reploc            INT   changement de repere
c9......................................................................
c10   commons utilises
c10
c10   planet                  caracteristqiues planete
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  xvabsl (xposit,xvites,
     +                    posita,vitesa)
c
      implicit none
c
      integer i
c
      double precision  xposit(3),xvites(3),posita(3),vitesa(3),
     +                  plocal(3,3),requat,rpolar,vitese(3),vitesl(3),
     +                  vitesr(3),xomega
c
      common / planet / xomega(3),requat,rpolar
c
c		passages en coordonnees cartesiennes
c
      call  cartes (xposit,0,
     +              posita)
c
c		calcul de la vitesse relative repere geocentrique
c
      call  cartes (xvites,1,
     +              vitesl)

      call  reploc (xposit,0,
     +              plocal)

      call  matvec (plocal,vitesl,3,3,
     +              vitesr)
c
c		vitesse d'entrainement
c
      call  pvecto (xomega,posita,
     +              vitese)
c
c		vitesse absolue
c
      do  i = 1,3
          vitesa(i) = vitese(i) + vitesr(i)
      end do
c
      return
      end
