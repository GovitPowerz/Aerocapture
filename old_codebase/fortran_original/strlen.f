c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : strlen.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine la longueur d'une chaine de caracteres
c3
c3    NOTA  la longueur d'une chaine de caracteres est obtenue lorsqu'on
c3          trouve un caractere blanc dans la chaine (au maximum, 72 ca-
c3          racteres)
c3......................................................................
c4    variables d'entree
c4
c4    chaine            A72   chaine de caracteres
c4......................................................................
c6    variables de sortie
c6
c6    strlen            I4    longueur de la chaine de caracteres
c6......................................................................
c8    composants appelants
c8
c8    entree            INT   choix des conditions de simulation
c8    opnfic            INT   ouverture des ficheirs resultats
c8......................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      function  strlen (chaine)
c
      implicit none
c
      integer  strlen,
     +         i,ichain
c
      character *1   a
      character *72  chaine
c
      i      = 0
      ichain = 0
c
      do  while (ichain.eq.0)
          i = i + 1
          a = chaine(i:i)
          if (a.eq.' ') then
             ichain = 1
             strlen = i-1
          endif
          if (i.eq.72) then
             ichain = 1
             strlen = 72
          endif
      end do
c
      return
      end
