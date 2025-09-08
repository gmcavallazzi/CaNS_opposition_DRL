FC = gfortran
FFLAGS = -O2 -g

neural_network: neural_network.f90
	$(FC) $(FFLAGS) -o neural_network neural_network.f90

neural_network_text: neural_network_text.f90
	$(FC) $(FFLAGS) -o neural_network_text neural_network_text.f90

neural_network_simple: neural_network_simple.f90
	$(FC) $(FFLAGS) -o neural_network_simple neural_network_simple.f90

clean:
	rm -f neural_network neural_network_text neural_network_simple *.o *.mod

.PHONY: clean