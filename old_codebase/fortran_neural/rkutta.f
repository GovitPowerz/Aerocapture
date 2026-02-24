c1
c1    copyright (c) AEROSPATIALE 1993
c1......................................................................
c2    nom    : rkutta.f
c2    date   : 18/08/93
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module realise l'integration numerique d'un systeme du premier
c3    ordre (mis sous la forme dx/dt = a.x + b) par la methode de Runge-
c3    Kutta d'ordre 4.
c3
c3    nota: La taille du vecteur x est limitee a 30 composantes.
c3......................................................................
c4    variables d'entree
c4
c4    xderiv            R8    etat a integrer
c4    pasint            R8    pas d'integration
c4    increm            I4    increment du Runge-Kutta (de 1 a 4)
c4    ndimxd            R8    dimension de la variable a integrer
c4    ix                I4    variable interne
c4......................................................................
c5    variables d'entree-sortie
c5
c5    qk                R8
c5......................................................................
c6    variables de sortie
c6
c6    xintgr            R8     etat integre
c6......................................................................
c11   norme logicielle GENE S320
c11
c11.....................................................................
c
c
      subroutine  rkutta  (pasint,xderiv,increm,ndimxd,ix,
     +                     qk,
     +                     xintgr)
c
c
      implicit none
      integer  i,ix,increm,ndimxd
c
      real*8  a,pasint,xderiv(ndimxd),xintgr(ndimxd),qk(30),xk(30)
c
c           initialisation
c
      a = sqrt(2.d0)
      do  i = 1,ndimxd
          xk(i) = pasint*xderiv(i)
      end do
c
c           test selon l'increment du Runge-Kutta
c
      if (increm.eq.1) then
         do  i = 1,ndimxd
             xintgr(i) = xintgr(i) + 0.5d0*xk(i)
             qk(i)      = xk(i)
         end do
         ix = 1
      else
         if ((increm.eq.2).or.(increm.eq.3)) then
            do  i = 1,ndimxd
                xintgr(i) = xintgr(i) + (1.d0 - ix/a)*(xk(i) - qk(i))
                qk(i)     = qk(i)*(-2.d0 + 3*ix/a) +
     +                      xk(i)*( 2.d0 - ix*a)
            end do
            ix =-1
         else
            do  i = 1,ndimxd
                xintgr(i) = xintgr(i) + (xk(i) - 2.d0*qk(i))/6.d0
            end do
         endif
      endif
c
      return
      end
