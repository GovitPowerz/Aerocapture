      SUBROUTINE interp(valrxx,tablxx,tably1,tably2,tably3,
     &                  valry1,valry2,valry3,npoint)

      
      real*8 valrxx,deltax,tablxx(npoint),
     *       tably1(npoint),tably2(npoint),tably3(npoint)
      
      real*8 valry1,valry2,valry3
      
      integer ienerg,jenerg,npoint
      logical test


C    POSITIONNEMENT TABLE ALTITUDE
C    *****************************

C     ENERGIE SUPERIEURE A tablxx(ienerg)

      ienerg = 2
      
      write(6,*) 'valrxx = ',valrxx
      
      
      IF (valrxx.GT.tablxx(ienerg)) THEN
        ienerg = ienerg - 1
        test = .true.
        write(6,*) 'valrxx > tablxx(',ienerg,')'
	do while (test )
             ienerg = ienerg + 1
             test = (ienerg.LT.npoint.AND.valrxx.GT.tablxx(ienerg))
	     write(6,*)inerg,npoint 
	     write(6,*)valrxx,tablxx(ienerg) 
        enddo
      ELSE
        ienerg = ienerg + 1

C        ENERGIE INFERIEURE A tablxx(ienerg)

        test = .true.
        write(6,*) 'valrxx <= tablxx(',ienerg-1,')'
        do while (test )
         ienerg = ienerg - 1
         test = (ienerg.GT.2.AND.valrxx.LT.tablxx(ienerg-1)) 
	 write(6,*) ienerg,valrxx,tablxx(ienerg-1)
        enddo
      END IF
      write(6,*) 'ienerg choisi = ',ienerg

C    PENTE ( ECART valrxx - tablxx )
C    ***************************
      jenerg = ienerg - 1
      if(jenerg .le. 0)jenerg = 1
      deltax = valrxx - tablxx(jenerg)
      
      write(6,*) jenerg,deltax,tably1(jenerg)
      
      valry1 = tably1(jenerg) +
     +        (deltax*(tably1(jenerg+1) - tably1(jenerg)))/
     +                (tablxx(jenerg+1) - tablxx(jenerg))
     
      write(6,*)  'valry1',valry1     
      
      valry2 = tably2(jenerg) +
     +        (deltax*(tably2(jenerg+1) - tably2(jenerg)))/
     +                (tablxx(jenerg+1) - tablxx(jenerg))
     
      write(6,*)  'valry2',valry2     
      
      valry3 = tably3(jenerg) +
     +        (deltax*(tably3(jenerg+1) - tably3(jenerg)))/
     +                (tablxx(jenerg+1) - tablxx(jenerg))
     
      write(6,*)  'valry3',valry3     
      
      
      return 
      end
