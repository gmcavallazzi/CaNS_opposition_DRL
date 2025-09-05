FC = mpifort
FFLAGS = -O2

all: parent child intensive_parent intensive_child persistent_workers simple_spawn simple_child simple_persistent

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

simple_spawn: simple_spawn.f90
	$(FC) $(FFLAGS) -o simple_spawn simple_spawn.f90

simple_child: simple_child.f90
	$(FC) $(FFLAGS) -o simple_child simple_child.f90

simple_persistent: simple_persistent.f90
	$(FC) $(FFLAGS) -o simple_persistent simple_persistent.f90

clean:
	rm -f parent child intensive_parent intensive_child persistent_workers simple_spawn simple_child simple_persistent *.mod

.PHONY: all clean