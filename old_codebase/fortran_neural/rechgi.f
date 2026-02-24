
	subroutine  rechgi(icarlo,nbsimu,sufdin)
	
	common / recher / pente
	
	character  *72 sufdin

	integer reussi,nbsimu,icarlo
	
	double precision  finite,energi,enrvis,interv
	
	double precision  intmax,intmin,pente
	
	interv = 45.
	finite = 0.0000000001
	intmax = 0.
	intmin = -90.
	pente = -45.
		
c
c		conditions generales de simulation
c
	close(unit=5)
	open(unit=5,file='aerocap'//sufdin,form='formatted')
	call  cisimu (icarlo,nbsimu)
c
c		simulation de l'aerocapture
c
	call simmsr (icarlo,nbsimu)
		
	open(unit=280,file='../sorties/gite.90',form='formatted')

	do while (interv.ge.finite)
		
		open(unit=260,file='../sorties/energie.finale',form='formatted')
		read(260,6000) energi,reussi,enrvis
		close(unit=260)
	
		write(280,4000) pente,enrvis,energi,interv

		if (energi.le.enrvis) then
			write(6,*) 'au dessus'
			intmin = pente
			interv = interv/2.
			pente = intmin + interv		
		else
			write(6,*) 'en dessous'
			intmax = pente
			interv = interv/2.
			pente = intmin + interv	
		endif
c
c		conditions generales de simulation
c
		close(unit=5)
		open(unit=5,file='aerocap'//sufdin,form='formatted')
		call  cisimu (icarlo,nbsimu)
c
c		simulation de l'aerocapture
c
		call simmsr (icarlo,nbsimu)
	end do
	
	write(6,*) pente

 4000 format(4(1x,d20.10))
 6000 format(1x,d20.10,1x,I1,1x,d20.10)

	return
	end

