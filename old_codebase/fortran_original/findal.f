c
	subroutine findal (precur,predyn,altitu,nbrval,altcor)
	
	integer nbrval,i
	
	double precision predyn(10000),altitu(10000)
	
	double precision precur,ecarpd,altcor(2)
	
	ecarpd = 10000
	chgsgn = 0
	altcor(1)=0.
	altcor(2)=0.
	i=1
	
	do while ((precur-predyn(i)).gt.0)
		
		i = i+1

	end do
	
	if (dabs(precur-predyn(i-1)).lt.dabs(precur-predyn(i))) then
		altcor(1) = altitu(i-1)
	else
		altcor(1) = altitu(i)
	endif
	
	do while ((precur-predyn(i)).lt.0)
		
		i = i+1

	end do
	
	if (dabs(precur-predyn(i-1)).lt.dabs(precur-predyn(i))) then
		altcor(2) = altitu(i-1)
	else
		altcor(2) = altitu(i)
	endif
	
	return
	end 
	