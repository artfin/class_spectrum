.PHONY: clean all 

CCXX = g++
MPICCXX = mpic++

CXXFLAGS = -std=c++11 -O2 -g -lstdc++ -lm

BUILDDIR := obj/
SOURCE_DIRS := ./DIP/ ./GEAR/ ./MISC/ ./POT/ ./he_ar/ 
VPATH := $(SOURCE_DIRS)

##############################################
uname_n := $(shell uname -n)
ifeq ($(uname_n),artfin-MacBookPro)
	hep_path:= /home/artfin/Downloads/hep-mc-0.5/include/
else
	hep_path := /n/home11/artfin/hep-mc-0.5/include/	
endif

eigen_path := /usr/local/include/eigen3
gnuplot_io_path := /home/artfin/Desktop/repos/gnuplot-iostream

st_libs := $(hep_path) $(eigen_path) $(gnuplot_io_path)
st_libs := $(addprefix -I , $(st_libs))

link_fftw3 := -lfftw3
link_gsl := -lgsl -lgslcblas
ld_libs := $(link_fftw3) $(link_gsl) 
##############################################

#########################################################################
#
# Gear files 
#
gear_diatom_src := awp.cpp basis_r.cpp fgauss.cpp gear_diatom.cpp vmblock.cpp
gear_diatom_obj := $(addprefix $(BUILDDIR), $(patsubst %.cpp, %.o, $(gear_diatom_src)))
#########################################################################

#########################################################################
# He - Ar programs
#
# mcmc_he_ar_src := $(foreach filename,$(mcmc_he_ar_src),$(shell find -type f -name $(filename)))

HE_AR_TARGETS := mcmc_he_ar grid_he_ar

mcmc_he_ar_src := matrix_he_ar.cpp mcmc_generator.cpp parameters.cpp file.cpp ar_he_dip_buryak_fit.cpp ar_he_pes.cpp ar_he_pes_derivative.cpp leg_arr.cpp fft.cpp 
mcmc_he_ar_srcx := spectrum_info.cc mcmc_he_ar.cc 
mcmc_he_ar_obj := $(addprefix $(BUILDDIR), $(patsubst %.cpp, %.o, $(mcmc_he_ar_src))) 
mcmc_he_ar_objx := $(addprefix $(BUILDDIR), $(patsubst %.cc, %.oo, $(mcmc_he_ar_srcx)))
mcmc_he_ar: $(mcmc_he_ar_obj) $(mcmc_he_ar_objx) $(gear_diatom_obj)
	@echo "(MCMC_HE_AR) object files: " $(mcmc_he_ar_obj)
	@$(MPICCXX) $(CXXFLAGS) $(ld_libs) $^ -o $@

grid_he_ar_src := $(mcmc_he_ar_src)
grid_he_ar_srcx := spectrum_info.cc grid_he_ar.cc
grid_he_ar_obj := $(addprefix $(BUILDDIR), $(patsubst %.cpp, %.o, $(grid_he_ar_src))) 
grid_he_ar_objx := $(addprefix $(BUILDDIR), $(patsubst %.cc, %.oo, $(grid_he_ar_srcx)))
grid_he_ar: $(grid_he_ar_obj) $(grid_he_ar_objx) $(gear_diatom_obj)
	@echo "(GRID_HE_AR) object files: " $(grid_he_ar_obj) $(grid_he_ar_objx)
	@$(MPICCXX) $(CXXFLAGS) $(ld_libs) $^ -o $@
#########################################################################

#########################################################################
# $(@D) -- directory where current target $@ resides in
#
$(BUILDDIR)%.o: %.cpp 
	@mkdir -p $(@D)	
	@$(CCXX) $(CXXFLAGS) -c -MD $(st_libs) $(ld_libs) $(addprefix -I,$(SOURCE_DIRS)) $< -o $@
	@echo ">> (g++) COMPILING $<...";

$(BUILDDIR)%.oo: %.cc
	@mkdir -p $(@D)	
	@$(MPICCXX) $(CXXFLAGS) -c -MD $(st_libs) $(ld_libs) $(addprefix -I,$(SOURCE_DIRS)) $< -o $@
	@echo ">> (mpic++) COMPILING $<...";

include $(wildcard *.d)
#########################################################################

targets := $(HE_AR_TARGETS) 

all: $(targets)

clean:
	@echo "Cleaning $(BUILDDIR) directory..."
	@rm -f $(BUILDDIR)*.o 
	@rm -f $(BUILDDIR)*.d
	@echo "Deleting executrables..."
	@rm -f $(targets)	
