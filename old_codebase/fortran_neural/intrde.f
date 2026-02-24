c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : intrmo.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module realise une interpolation lineaire dans une table monodi
c3    mensionnelle
c3
c3......................................................................
c4    variables d'entree
c4
c4    valrxx            R8    valeur courante
c4    tablxx (npoint)   R8    table des abscisses
c4    tablyy (npoint)   R8    table des ordonnees
c4    npoint            I4    nombre de points des tables
c4......................................................................
c5    variables d'entree-sortie
c5
c5    kinter            I4    pas d'interpolation
c5......................................................................
c6    variables de sortie
c6
c6    valryy            R8    parametre interpole
c6
c6......................................................................
c8    composants appelants
c8
c8    faeros            INT   coefficients aerodynamiques
c8    fatmos            INT   coefficients atmospheriques
c8......................................................................
c11   norme logicielle GENE S320
c11
c11   non               presence de goto
c11.....................................................................
c
      subroutine  intrde (valrxx,tablxx,tablyy,npoint,
     +                    kinter,
     +                    valryy)
c
      implicit none
c
      integer  npoint,i,k,kinter
c
      double precision  valrxx,valryy,tablxx(npoint),tablyy(npoint)
c
      k = kinter
      
c
      do  i = 1,npoint
c
	 if ((valrxx.ge.tablxx(k)).and.
     +       (valrxx.lt.tablxx(k-1))) then
c
            valryy = tablyy(k-1) +
     +              ((valrxx - tablxx(k-1))*(tablyy(k) - tablyy(k-1)))/
     +                                      (tablxx(k) - tablxx(k-1))
            goto 10
         else
            if (valrxx.lt.tablxx(k)) then
               if (k.eq.npoint) then
                   valryy = tablyy(npoint)
                   goto 10
               else
                   k = k + 1
               endif
            else
               if (k.eq.2) then
                   valryy = tablyy(1)
                   goto 10
               else
                   k = k - 1
               endif
            endif
         endif
c
      end do
c
 10   continue
c
      kinter = k
c
      return
      end
