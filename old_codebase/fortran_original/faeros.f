c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : faeros.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine les coefficeints aerodynamiques de la capsule
c3    (soit Cx et Cz) en fonction du nombre de Mach et de l'incidence
c3    (on fait l'hypothese d'un vol equilibre)
c3
c3......................................................................
c4    variables d'entree
c4
c4    xincid            R8    incidence
c4......................................................................
c5    variables d'entree-sortie
c5
c5    kintar            I4    increment interpolation tables atmosphere
c5......................................................................
c6    variables de sortie
c6
c6    vitmac            R8    nombre de Mach
c6    cxcaps            R8    coefficient de trainee
c6    czcaps            R8    coefficient de portance
c6......................................................................
c7    variables internes
c7
c7    vitson            R8    vitesse du son
c7......................................................................
c8    composants appelants
c8
c8    realit            INT   integration trajectoire reelle
c8    trajec            INT   prediction de trajectoire
c8......................................................................
c9    composants appeles
c9
c9    intrmo            INT   intepolation lineaire monodimensionnelle
c9......................................................................
c10   commons utilises
c10
c10   tabaer                  tables aerodynamiques
c10   tablar                  nombre de parametres des tables aeros
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   non                   parametres variables common / intatm /
c11.....................................................................
c
      subroutine  faeros (xincid,
     +                    kintar,
     +                    cxcaps,czcaps)
c
      implicit none
c
      include '../include/dimensions.incl'
c
      integer  kintar,
     +         kintcx,kintcz,nbmach
c
      double precision  xincid,cxcaps,czcaps,
     +                  tabmac,tabcxe,tabcze
c
      common / tabaer / tabmac(nmachx),tabcxe(nmachx),tabcze(nmachx)
      common / tablar / nbmach
c
      common / intaer / kintcx,kintcz
c
c		interpolation des tables aerodynamiques
c
      kintcx = kintar
      kintcz = kintar
c
      call  intrmo (xincid,tabmac,tabcxe,nbmach,
     +              kintcx,
     +              cxcaps)
      call  intrmo (xincid,tabmac,tabcze,nbmach,
     +              kintcz,
     +              czcaps)
c
      kintar = kintcz
c
      return
      end
