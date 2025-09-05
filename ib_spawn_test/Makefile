FC = mpifort
FFLAGS = -O2

all: parent child intensive_parent intensive_child persistent_workers

parent: parent.f90
	$(FC) $(FFLAGS) -o parent parent.f90

child: child.f90
	$(FC) $(FFLAGS) -o child child.f90

intensive_parent: intensive_parent.f90
	$(FC) $(FFLAGS) -o intensive_parent intensive_parent.f90

intensive_child: intensive_child.f90
	$(FC) $(FFLAGS) -o intensive_child intensive_child.f90

persistent_workers: persistent_workers.f90
	$(FC) $(FFLAGS) -o persistent_workers persistent_workers.f90

clean:
	rm -f parent child intensive_parent intensive_child persistent_workers *.mod

.PHONY: all clean