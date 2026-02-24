c1
c1    copyright (c) eads launch vehicles 2002
c1......................................................................
c2    nom    : lecaer.f
c2    date   : 01/07/02
c2    iv     : 1
c2    ie     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module lit les caracteristiques du vehicule:
c3    - masse
c3    - surface de reference
c3    - tables CA, Cn fonction de (Mach, incidnce)
c3    - tables de dispersions Ca et Cn fonction du Mach
c3    - table d'evolution de l'incidence d'equilibre fonction du Mach
c3    - plage de meconnaissance de l'incidence d'equilibre
c3
c3    Il est possible de definir les coefficients aerodynamqiues Cx et
c3    Cz au moyen de polaires (indicateur indpol).
c3    Dans le cas de la simualtion de la phase TYAEM (natgui = 4 ou 5),
c3    les coefficients Ca et Cn sont transformes en Cx-Cz pour les be-
c3    soins du guidage en phase TAEM
c3......................................................................
c8    composants appelants
c8
c8    lecdat              appelle tous les modules de lecture
c8......................................................................
c10   commons utilises
c10
c10   modgui             type de guidage
c10   trigon             contantes trigonometriques
c10
c10   aeroi              table aerodynamique
c10   tvmach(naero) R8   nombre de mach
c10   tcx(naero)    R8   valeur du Cxe
c10   tcz(naero)    R8   valeur du Cze
c10
c10   cacn0
c10   xmacn(nbmac)  R8    points de Mach pour les table Ca-Cn
c10   xalfa(nbmac)  R8    points d'incidence pour les tables Ca-Cn
c10   ca0(nbmac,    R8    coefficients de trainee
c10       nalfa)
c10   cn0(nbmac,    R8    coefficients de portance
c10       nalfa)
c10   dca(nbmac)    R8    meconnaissance sur Ca fonction du Mach
c10   dcn(nbmac)    R8    meconnassiance sur Cn fonction du Mach
c10
c10   disaeq
c10   aeqmin        R8    borne inf de variation de l'incidence
c10   aeqmax        R8    borne sup de variation de l'incidence
c10   aeqpen        R8    variation de la pente
c10   aeqori        R8
c10
c10
c10   equil
c10   alfat(nbmac)  R8    table de variation de l'incidence d'equilibre
c10                       selon le nombre de Mach
c10
c10   modalf             modelde d'incidence
c10   modgui             modele de guidage
c10   trigon             constantes trigonometriques
c10
c10   modpol
c10   indpol       I4    indicateur de poliares Cx-Cz equilibrees
c10
c10   polaer
c10   coefcx(3)    R8    coefficeint de la polaire Cx = f(alfa)
c10   coefcz(3)    R8    coefficeint de la polaire Cz = f(alfa)
c10
c10   sizaer
c10   nalfa        I4    nombre de points d'incidence tables aeros
c10   nbmac        I4    nombre de points de Mach tables aeros
c10   nbmaca       I4    nombre de points de Mach BO/equilibre
c10   naero        I4    nombre de points d'incidence BO/equilibre
c10.....................................................................
c11   includes utilises
c11
c11   dim_tables         dimension des tables de donnees
c11.....................................................................
c13   norme logicielle gene s320
c13
c13   non              presence de "include", "stop"
c13.....................................................................
c
      subroutine lecgnn
c
      implicit none
c
      include '../donnees/param_algo'
c
      integer  i,j
c
      read(923,*)
      read(923,*)
      read(923,*)
      read(923,*)
      read(923,*)
      read(923,*)
c
      do i = 1,n1input
         do j = 1,n1hid1
            read(923,*) n1lw1(j,i)
         enddo
      enddo
      do i = 1,n1hid1
         read(923,*) n1bias1(i)
      enddo
c
      do i = 1,n1hid1
         do j = 1,n1output
            read(923,*) n1lw4(j,i)
         enddo
      enddo
      do i = 1,n1output
         read(923,*) n1bias4(i)
      enddo
c
      return
      end
