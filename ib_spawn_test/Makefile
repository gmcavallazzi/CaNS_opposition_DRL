FC = mpifort
FFLAGS = -O2

all: simple_spawn simple_child simple_persistent

simple_spawn: simple_spawn.f90
	$(FC) $(FFLAGS) -o simple_spawn simple_spawn.f90

simple_child: simple_child.f90
	$(FC) $(FFLAGS) -o simple_child simple_child.f90

simple_persistent: simple_persistent.f90
	$(FC) $(FFLAGS) -o simple_persistent simple_persistent.f90

clean:
	rm -f simple_spawn simple_child simple_persistent *.mod *.log hostfile*

.PHONY: all clean