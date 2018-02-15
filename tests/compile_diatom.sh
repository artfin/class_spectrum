#!/bin/sh
g++ -std=c++11 full_diatom_distribution.cpp -o full_diatom_distribution -I /usr/local/include/eigen3 ../obj/mcmc_generator.o ../obj/file.o ../obj/parameters.o ../obj/ar_he_pes.o -lgsl -lgslcblas
