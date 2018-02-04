INCLUDE_EIGEN = -I /usr/local/include/eigen3
INCLUDE_HEP = -I /home/artfin/Downloads/hep-mc-0.5/include/

test: test.cpp
	mpic++ -std=c++11 $(INCLUDE_EIGEN) $(INCLUDE_HEP) ../obj/ar_he_pes.o $< -o $@
