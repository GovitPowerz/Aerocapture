c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : pilote.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module simule le fonctionnement du pilote. A partir de la gite
c3    commandee par le guidage, il fournit la gite realisee en tenant
c3    compte d'une modelisation adaptee du pilote (1er ou 2nd ordre, pri
c3    se en compte d'une vitesse de roulis maximale...).
c3
c3    NOTA  Actuellement, on fait l'hypothese d'un pilote parfait.
c3
c3......................................................................
c4    variables d'entree
c4
c4    gitcom            R8    gite commandee par le guidage        (rad)
c4    vitcom            R8    vitesse de gite commandee avant saturation
c4......................................................................
c6    variables de sortie
c6
c6    gitpil            R8    gite realisee par le pilote          (rad)
c6    vitpil            R8    vitesse de gite                    (rad/s)
c6......................................................................
c8    composants appelants
c8
c8    simmsr            INT  simulation de l'aerocapture
c8......................................................................
c10   commons utilises
c10
c10   carpil
c10   modpil
c10   period
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  pilote (positn,vitesn,alfcom,gitcom,gpilpr,
     +                    vitcom,datpil,
     +                    alfpil,gitpil,vitpil)
c
      implicit none
      
      integer  natpil
c
      double precision  positn(3),vitesn(3),alfcom,gitcom,vitcom,
     +                  alfpil,gitpil,vitpil,cstpil,amrpil,omgpil,
     +                  datpil,gpilpr
     
      common / carpil / cstpil,amrpil,omgpil
      common / modpil / natpil
      
      intrinsic  dexp,dsqrt
c
c		hypothese de pilote parfait
c
      if (natpil.eq.0) then
         alfpil = alfcom
         gitpil = gitcom
         vitpil = vitcom
      endif
c
c		hypothese de pilote selon 1er ordre (sur gite)
c
      if (natpil.eq.1) then
         alfpil = alfcom
         gitpil = gpilpr + 
     +           (gitcom - gpilpr)*(1.d0 - dexp(-datpil/cstpil)) 
         vitpil = vitcom
      endif
c
      return
      end
