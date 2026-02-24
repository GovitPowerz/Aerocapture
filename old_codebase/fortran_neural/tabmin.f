c1
c1    copyright (c) AEROSPATIALE 2000
c1......................................................................
c2    nom    : tabmin.f
c2    date   : 13/04/00
c2    IV     : 1
c2    IE     : 1
c2    auteur : Mechin D.
c2......................................................................
c3    Ce module permet de construire, a partir de la table donnant
c3    (energie totale - vit. radiale - Pdyn) sur la traj.min., la
c3    table correspondante mais avec une "base d'energie totale" a pas
c3    constant.
c3
c3......................................................................
c4    variables d'entree
c4
c4......................................................................
c5    variables d'entree-sortie
c5
c5......................................................................
c6    variables de sortie
c6
c6......................................................................
c7    variables internes
c7
c7......................................................................
c8    composants appelants
c8
c8......................................................................
c9    composants appeles
c9
c9    intrmo           INT    guidage en incidence
c9......................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine tabmin (enrjmi,vitrmi)
c
      implicit none
c
      integer  i,kintop,ntabla,ntabul,p
c
      double precision  denrjl,enrjlt,enrjmi,pdynmi,pdymin,vitmin,vitrmi
     +                  
c
c		initialisations
c
      enrjlt = 0.d0
c
c		generation de la table (energie totale - vit. radiale - Pdyn)
c
      do  p = 1,503

         enrjlt = enrjlt + denrjl
                           
         call  intrmo (enrjlt,enrjmi,vitrmi,ntabul,
     +                 kintop,
     +                 vitmin)
         call  intrmo (enrjlt,enrjmi,pdynmi,ntabul,
     +                 kintop,
     +                 pdymin)
     
         write(201,*) enrjlt,vitmin,pdymin
          
      enddo
 
      return
      end
      
      
      
