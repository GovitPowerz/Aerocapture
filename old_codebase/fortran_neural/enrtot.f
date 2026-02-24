c1
c1    copyright (c) AEROSPATIALE 1999
c1......................................................................
c2    nom    : enrtot.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine la valeur de l'energie totale (i.e. somme de
c3    de l'energie cinetique et de l'energie potentielle) calculee a
c3    partir des positions et vitesses absolues.
c3......................................................................
c4    variables d'entree
c4
c4    xposit(3)         R8    position absolue geocentrique spherique
c4    xvites(3)         R8    vitesse relative locale spherique
c4    varia             I4    nom
c4......................................................................
c6    variables de sortie
c6
c6    enrtot            R8    valeur courante de l'abscisse
c6......................................................................
c7    variables internes
c7
c7    rayvec            R8    altitude geocentrique
c7    vitabs            R8    norme de la vitesse absolue
c7......................................................................
c8    composants appelants
c8
c8    carltf            INT   parametres fin de simulation
c8    carltz            INT   parametres debut de simulation
c8    energi            INT   parametres energetiques
c8    etafin            INT   edition ecran conditions finales
c8......................................................................
c9    composants appeles
c9
c9    pnorme            INT   norme de vecteur
c9    xvabsl            INT   position-vitesse absolues
c9......................................................................
c10   commons utilises
c10
c10   geoide                  caracteristiques champ de pesanteur
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      function  enrtot (xposit,xvites)
c
      implicit none
c
      double precision  xposit(3),xvites(3),enrtot,
     +                  excent,posita(3),rayvec,vitabs,vitesa(3),xj2,
     +                  xmug,
     +                  pnorme
c
      common / geoide / excent,xj2,xmug
c
      external  pnorme
c
c		position et vitesse absolue
c
      call  xvabsl (xposit,xvites,
     +              posita,vitesa)
c
c		energie
c
      vitabs = pnorme(vitesa)
      rayvec = pnorme(posita)
c
      enrtot = vitabs**2/2.d0 - xmug/rayvec
c
      return
      end
