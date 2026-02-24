c1
c1    copyright (c) AEROSPATIALE 2000
c1......................................................................
c2    nom    : tabula.f
c2    date   : 13/04/00
c2    IV     : 1
c2    IE     : 1
c2    auteur : Mechin D.
c2......................................................................
c3    Ce module permet de construire, a partir des 3 tables donnant
c3    (energie totale - vit. radiale - Pdyn) 3 tables correspondantes
c3    ayant une "base d'energie" identique.
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
      program tabula
c
      implicit none
c
      integer  i,j,k,ntabla,ntablb,ntablc,p
     +         
c
      double precision  denrjl,enrjlt,enrjop
     +                  
c
c		ouverture des fichiers de donnees	
c
      open(unit= 101, file= '../donnees/resu_pilo0',
     +                form= 'formatted')
      open(unit= 102, file= '../donnees/resu_pilo90',
     +                form= 'formatted')     
      open(unit= 103, file= '../donnees/resu_pilo180',
     +                form= 'formatted')      
     
c
c		ouverture des fichiers de resultats	
c
      open(unit= 201, file= '../donnees/resu_pilo_new0',
     +                form= 'formatted')
      open(unit= 202, file= '../donnees/resu_pilo_new90',
     +                form= 'formatted')     
      open(unit= 203, file= '../donnees/resu_pilo_new180',
     +                form= 'formatted')      
     
c
c		lecture des fichiers de donnees
c      
      read(101,*) ntabla
      do  i = 1,ntabla
          read(101,*) enrjmi(i),vitrmi(i),pdyopt(i),gainhr(i),gainpr(i)
      end do
            
      read(102,*) ntablb
      do  j = 1,ntablb
          read(102,*) enrjop(i),vitopt(i),pdyopt(i),gainhr(i),gainpr(i)
      end do
            
      read(103,*) ntablc
      do  k = 1,ntablc
          read(103,*) enrjop(i),vitopt(i),pdyopt(i),gainhr(i),gainpr(i)
      end do

c
c		fermeture des fichiers de donnees
c
      close(unit= 101)
      close(unit= 102)
      close(unit= 103)
      
c
c		creation des nouvelles tables
c
      call  tabref (enrre,vitrre)
      call  tabmin (enrjmi,vitrmi)
      call  tabmax (enrjma,vitrma)

      stop
      end
      
      
      
