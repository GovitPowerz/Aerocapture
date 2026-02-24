c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : carlts.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module realise la sauvegarde des conditions initiales en cas de
c3    Monte-Carlo (sauvegarde sur fichier formatte a acces sequentiel).
c3
c3    NOTA  le signe - sur l'azimut de la vitesse est du a la convention
c3          de signe adoptee par MAXTOM
c3......................................................................
c4    variables d'entree
c4
c4    xposit(3)         R8    position reelle repere geocentrique
c4    xvites(3)         R8    vitesse reelle repere local
c4    numero            I4    numero de simulation courant
c4......................................................................
c7    variables internes
c7
c7    altitu            R8    altitude courante
c7    latitu            R8    latitude courante
c7    longit            R8    longitude courante
c7    rayvec            R8    rayon vecteur courant
c7    vitess            R8    norme de la vitesse relative
c7    vitrad            R8    vitesse radiale
c7    xenerg            R8    energie totale
c7    xrayon            R8    rayon planete courant
c7......................................................................
c8    composants appelants
c8
c8    simmsr            INT   simulation aerocapture
c8......................................................................
c9    composants appeles
c9
c9    frayon            INT   rayon planete
c9    enrtot            INT   abscisse du profil de gite commandee
c9......................................................................
c10   commons utilises
c10
c10   geoide                  caracteristqiues champ de pesanteur
c10   trigon                  parametres trigonometriques
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  carltm (positr,vitesr,positn,vitesn)
c
      implicit none
c
      integer  k
c
      double precision  positr(3),vitesr(3),positn(3),vitesn(3),
     +                  degrad,pi,xsauve(6),cinexi,altitr,xlatit
c
      common / capexi / cinexi(6)
      common / trigon / degrad,pi
c
      call  frayon (positr,
     +              altitr,xlatit)
c
c		sauvegarde
c
      xsauve(1) = (altitr    - cinexi(1))/1.d3
      xsauve(2) = (positr(2) - cinexi(2))/degrad
      xsauve(3) = (xlatit    - cinexi(3))/degrad
      xsauve(4) = (vitesr(1) - cinexi(4))
      xsauve(5) = (vitesr(2) - cinexi(5))/degrad
      xsauve(6) = (vitesr(3) - cinexi(6))/degrad
c
      write(350,1000) (xsauve(k), k = 1,6)
c
 1000 format(6(1x,d12.5))
c
      return
      end
