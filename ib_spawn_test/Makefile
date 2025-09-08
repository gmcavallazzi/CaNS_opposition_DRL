FC = mpifort
FFLAGS = -O2

all: fortran_worker

fortran_worker: fortran_worker.f90
	$(FC) $(FFLAGS) -o fortran_worker fortran_worker.f90

clean:
	rm -f fortran_worker *.mod *.log hostfile*

.PHONY: all clean