FC = mpifort
FFLAGS = -O2

all: parent child

parent: parent.f90
	$(FC) $(FFLAGS) -o parent parent.f90

child: child.f90
	$(FC) $(FFLAGS) -o child child.f90

clean:
	rm -f parent child *.mod

.PHONY: all clean