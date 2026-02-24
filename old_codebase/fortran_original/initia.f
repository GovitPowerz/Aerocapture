c
      subroutine  initia
      
      integer  ntabul,itable,nzapd,nzapdmx

      parameter (itable = 2000)
      
      parameter (nzapdmx = 1000)
      
      double precision  nrjval,refpre,refhdt,nrjtst,refhtt,
     +                  refincli,refdates,nrjpre,nrjhdt,
     +                  refcmu,altpdn,tabpda,tabpdb
      
      character  *72 sufaer,sufatm,sufdis,sufgui,sufinc,suflot,sufmis,
     +               sufmsr,sufnav,sufren,sufsuc,sufres

      common / ficdat / sufaer,sufatm,sufdis,sufgui,sufinc,suflot,
     +                  sufmis,sufmsr,sufnav,sufren,sufsuc
      common / ficres / sufres
      common / nbgain / ntabul
      common / tabnrj / nrjval(itable), nrjpre(itable),nrjhdt(itable),
     +                  nrjtst(itable)
      common / reftab / refpre(itable), refhdt(itable),refhtt(itable),
     +                  refincli(itable),refdates(itable),refcmu(itable)
      common / modpdn / nzapd
      common / varpdn / altpdn(nzapdmx),tabpda(nzapdmx),tabpdb(nzapdmx)
      
      close(202)
      
      open(unit= 202, file= '../sorties/divers'//sufres,
     +                      form= 'formatted')

      open(unit= 952, file= '../donnees/guidage'//sufgui,
     +                   form= 'formatted')
      open(unit= 953, file= '../donnees/tables_energie_gains'
     +                                    //sufgui,
     +                   form= 'formatted')


      read(952,*)
      read(952,*)
      read(952,*)
      read(952,*) 
      read(952,*) 
      read(952,*) 
      read(952,*) 
      read(952,*) 
      read(952,*) 
      read(952,*) 
      read(952,*) 
      read(952,*) 
      read(952,*) 
      read(952,*) 
      read(952,*) 
      read(952,*)
      read(952,*)
      read(952,*)
      read(952,*)
      read(952,*)
      read(952,*)
      read(952,*)
      read(952,*)
      read(952,*) 
      read(952,*) nzapd
      
      if (nzapd.gt.nzapdmx) then
         nzapd = nzapdmx
      endif
      
      do  i = 1,nzapd
          read(952,*) altpdn(i),tabpda(i)
          
          tabpdb(i) = 0.d0
      end do
c
c		lecture du profil de commande sur trajectoire optimale
c
         do  i = 1,100000
             read(953,*,end=999) nrjval(i),refpre(i),refhdt(i),
     +                           refhtt(i),refincli(i),refdates(i),
     +                           refcmu(i)
     
             nrjval(i) = nrjval(i)*1.d6
             refpre(i) = refpre(i)
             refhdt(i) = refhdt(i)
                          
             ntabul = i
             
         end do
         
 999     close(953)
      
      close(952)
 
      return
      end
