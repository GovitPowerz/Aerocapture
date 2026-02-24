c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : sortie.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module realise la sauvegarde des conditions cinematiques de pas
c3    sage en phase de sortie.
c3
c3......................................................................
c4    variables d'entree
c4
c4    positr(3)         R8    position absolue reelle courante
c4    vitesr(3)         R8    vitesse relative reelle courante
c4    alfcom            R8    incidence commandee courante
c4    gitpre            R8    gite commandee precedente
c4    temsim            R8    temps courant
c4    icarlo            I4    indicateur de Monte-Carlo
c4    isimul            I4    numero de simulation courante
c4......................................................................
c8    composants appelants
c8
c8    simmsr            INT   simulation d'aerocapture
c8......................................................................
c9    composants appeles
c9
c9    frayon            INT   determination rayon planete
c9......................................................................
c10   commons utilises
c10
c10   fensim
c10   modres
c10   trigon                  parametres trigonometriques
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  sortie (positr,vitesr,gitpre,temsim,alfcom,
     +                    icarlo,isimul)
c
      implicit none
c
      integer  icarlo,isimul,
     +         isauve,numsim,numvis
c
      double precision  positr(3),vitesr(3),alfcom,gitpre,temsim,
     +                  altitu,degrad,pi,xlatit
c
      common / fensim / numsim,numvis
      common / modres / isauve
      common / trigon / degrad,pi
c
      if ((numsim.eq.isimul).and.(isauve.eq.1)) then
c
         call  frayon (positr,
     +                 altitu,xlatit)
c     
         write(240,*)
         write(240,*) '   Conditions debut phase de sortie'
         write(240,*)
         write(240,*)  altitu/1.d3
         write(240,*)  positr(2)/degrad
         write(240,*)  xlatit/degrad
         write(240,*)  vitesr(1)
         write(240,*)  vitesr(2)/degrad
         write(240,*)  vitesr(3)/degrad
         write(240,*)  temsim
         write(240,*)  gitpre/degrad
         write(240,*)  alfcom/degrad
         write(240,*)
      endif
c
      if (icarlo.eq.1) then
c
         call  frayon (positr,
     +                 altitu,xlatit)
c      
         write(320,1000) isimul,
     +                   altitu/1.d3,positr(2)/degrad,
     +                   xlatit/degrad,vitesr(1),vitesr(2)/degrad,
     +                   vitesr(3)/degrad,gitpre/degrad,
     +                   alfcom/degrad,temsim              
c      
      endif
c
 1000 format(1x,i4,9(1x,d12.5))     
c
      return
      end
