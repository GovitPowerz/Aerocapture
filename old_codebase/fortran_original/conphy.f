c1
c1    copyright (c) AEROSPTIALE 1999
c1......................................................................
c2    nom    : conphy.f
c2    date   : 01/09/99
c2    IV     : 1
c2    IE     : 1
c2    auteur : Vernis P.
c2......................................................................
c3    Ce module determine les valeurs courantes du facteur de charge et
c3    du flux thermique conventionnel
c3
c3......................................................................
c4    variables d'entree
c4
c4    xposit(3)         R8    position absolue repere geocentrique
c4    xvites(3)         R8    vitesse relative repere local
c4    temsim            R8    temps courant
c4    imodel            I4    indicateur de modele utilise
c4......................................................................
c5    variables d'entree-sortie
c5
c5    incrar            I4    increment interpolation table aerodynamique
c5    incrat            I4    increment interpolation table atmosphere
c5......................................................................
c6    variables de sortie
c6
c6    coefan(2)         R8    coefficients aerordynamiques
c6    xcharg            R8    facteur de charge
c6    xflutr            R8    flux thremique conventionnel
c6    xpdyna            R8    pression dynamique
c6    acdrag            R8    acceleration de trainee
c6    aclift            R8    acceleration de portance
c6......................................................................
c7    variables internes
c7
c7    altitu            R8    altitude
c7    cxcaps            R8    coefficient de trainee
c7    czcaps            R8    coefficient de portance
c7    romver            R8    densite atmospherique
c7    vitrel            R8    vitesse relative
c7    xlatit            R8    latitude
c7    xrayon            R8    rayon courant de la planete
c7......................................................................
c8    composants appelants
c8
c8    naviga            R8    modelisation de la navigation
c8......................................................................
c9    composants appeles
c9
c9    faeros            INT   coefficients aerodynamiques
c9    fatmos            INT   coefficients atmospheriques
c9    frayon            INT   caracteristiques geoide
c9......................................................................
c10   commons utilises
c10
c10   capsul                  caracteristiques capsule
c10.....................................................................
c11   norme logicielle GENE S320
c11
c11   oui
c11.....................................................................
c
      subroutine  conphy (xposit,xvites,alfcom,temsim,imodel,
     +                    incrar,incrat,
     +                    coefan,xcharg,xflutr,xpdyna,acdrag,aclift)
c
      implicit none
c
      integer  imodel,incrar,incrat,
     +         inctar,inctat,indalt,atmvar
c
      double precision  xposit(3),xvites(3),alfcom,temsim,coefan(2),
     +                  xcharg,xflutr,xpdyna,acdrag,aclift,
     +                  altitu,cxcaps,czcaps,romver,srefer,vgitmx,
     +                  vitrel,vitmac,vitson,xlatit,xlongi,xmasse,
     +                  dalfae,disatm,dadrag,dnlift,xaltro,xgabro,
     +                  dispro,dxdrag,dxlift,cq,dxmass,positz(3),
     +                  vitesz(3),poscur(3),pi,
     +                  ampli,wavlen,y,poscar(3),positc(3),degrad
c
      common / capsul / srefer,vgitmx,xmasse
      common / mecaer / dalfae,disatm,dadrag,dnlift
      common / mecmas / dxmass
      common / profro / xaltro(5),xgabro(5,2)
      common / raynez / cq
      
      common / xvrent / positz,vitesz
      common / trigon / degrad,pi
      common / varhor / atmvar,ampli,wavlen
c
      intrinsic  dsqrt
c
      inctar = incrar
      inctat = incrat
      xlongi = xposit(2)
c
c		calculs preliminaires
c
      call  frayon (xposit,
     +              altitu,xlatit)
c
      vitrel = xvites(1)
c
c		modele d'atmsophere
c      
      call  fatmos (altitu,xlatit,xlongi,temsim,imodel,
     +              inctat,
     +              romver,vitson)
c
      if (imodel.eq.0) then
         indalt = 0
         dispro = 0.d0
         do  while (indalt.ne.4) 
             indalt = indalt + 1
             if ((altitu.ge.xaltro(indalt)).and.
     +           (altitu.lt.xaltro(indalt+1))) then
                dispro = xgabro(indalt+1,1)*altitu + xgabro(indalt+1,2)
             endif
         end do
         romver = romver*(1.d0 + dispro*disatm)
         if (atmvar.ge.1) then
          call geodes(altitu,xlatit,xlongi,poscur)
          call cartes(poscur,0,poscar)
          call cartes(positz,0,positc)
          y=(positc(1)-poscar(1))**2+(positc(2)-poscar(2))**2
          y=y+(positc(3)-poscar(3))**2-(poscur(1)-positz(1))**2
          y=sqrt(y)/1000.
          romver=romver*(1+ampli*(dsin(2*pi*y/wavlen+3*pi/4)))
         endif
      endif   
      vitmac = vitrel/vitson
c
c		modele aerodynamique
c
      call  faeros (alfcom,
     +              inctar,
     +              cxcaps,czcaps)
c
c		coefficients aerodynamiques
c
      if (imodel.eq.0) then
          alfcom = alfcom + dalfae
          dxdrag = dadrag*dcos(alfcom) + dnlift*dsin(alfcom)
          dxlift =-dadrag*dsin(alfcom) + dnlift*dcos(alfcom)      
          cxcaps = cxcaps*(1.d0 + dxdrag)
          czcaps = czcaps*(1.d0 + dxlift)      
      endif
      coefan(1) = cxcaps
      coefan(2) = czcaps
c
c		accelerations de trainee et de portance
c
      if (imodel.eq.1) then
         acdrag = romver*srefer*cxcaps*vitrel**2/(2.d0*xmasse)
         aclift = romver*srefer*czcaps*vitrel**2/(2.d0*xmasse)
      else
         acdrag = romver*srefer*cxcaps*vitrel**2/
     +           (2.d0*xmasse*(1.d0 + dxmass))
         aclift = romver*srefer*czcaps*vitrel**2/
     +           (2.d0*xmasse*(1.d0 + dxmass))      
      endif
c
c		facteur de charge
c
      xcharg = dsqrt(acdrag**2 + aclift**2)
c
c		flux thermique capsule type aérocapture
c
      xflutr = cq*dsqrt(romver)*vitrel**3.05
c
c		pression dynamique
c
      xpdyna = romver*vitrel**2/2.d0
c
      return
      end
