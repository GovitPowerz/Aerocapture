c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : .f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine les parametriques atmospheriques (densite et 
c3    vitesse du son) pour la planete Mars selon un modele exponentiel
c3    (pour utilisation par le guidage et la navigation) ou selon le mo
c3    dele MarsGram V3.8.
C3
C3    nota  Il est possible de sortir la pression atmospherique.
c3          Pour le modele embarque, on ne calcule pas (actuellement) la
c3          vitesse du son.
c3          Le module est prevu pour une date d'arrivee au 11 aout 2006
c3          a minuit. Toute modification de la date d'arrivee engendre
c3          une redefinition de la date d'arrivee.
c3          Les valeurs des coefficients CF0 ... sont extraites de l'ar
c3          ticle AIAA 98-4569 (D. Powell)
c3......................................................................
c4    variables d'entree
c4
c4    xaltit            R8    altitude geodesique
c4    xlatit            R8    latitude geodesique
c4    xlongi            R8    longitude geodesique
c4    temsim            R8    temps courant
c4    imodel            I4    nature du modele (embarque ou reel)
c4......................................................................
c5    variables d'entree-sortie
c5
c5    incrar            I4    indicateur de lecture dans les tables
c5......................................................................
c6    variables de sortie
c6
c6    romver            R8    densite atmospherique
c6    vitson            R8    vitesse du son
c6......................................................................
c7    variables internes
c7
c7    tarriv            R8    temps courant depuis l'arrivee (pour une 
c7                            date d'arrivee au 11 aout 2006 a minuit)
c7......................................................................
c8    composants appelants
c8
c8......................................................................
c9    composants appeles
c9
c9    atmos2            INT   modele MarsGram V3.8
c9......................................................................
c10   commons utilises
c10
c10   modatm                  modele d'atmospher exponentiel
c10   trigon                  parametres trigonometriques
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  fatmos (xaltit,xlatit,xlongi,temsim,imodel,
     +                    incrar,
     +                    romver,vitson)
c
      implicit none
c
      integer  imodel,incrar,
     +         naltit
c
      double precision  xaltit,xlatit,xlongi,temsim,romver,vitson,
     +                  cstgam,degrad,facech,pi,rozmod,rmoyen,
     +                  zromod,
     +                  altatm,romatm
c
      common / modatm / cstgam,facech,rozmod,rmoyen,zromod
      common / trigon / degrad,pi
      common / tabatm / altatm(1500),romatm(1500)
      common / nbzatm / naltit

c
      intrinsic  dble,sngl
c
      if (imodel.eq.1) then
c
c		modele d'atmosphere tabule sur Marsgram 3.8
c
         vitson = 1.d-33
c         
         call  intrmo (xaltit,altatm,romatm,naltit,
     +                 incrar,
     +                 romver)
c
      else
c
c		modele d'atmosphere
c
         vitson = 1.d-33
c         
         call  intrmo (xaltit,altatm,romatm,naltit,
     +                 incrar,
     +                 romver)
c      
      endif
c
      return
      end

