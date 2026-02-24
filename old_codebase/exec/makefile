# Unified Makefile for Aerocapture Fortran builds
# Builds both aerocap (original FTC) and aerocap_nn (neural network) variants

# Compiler selection
FC = gfortran

# Flags
DEBUG = -g -fbacktrace -fcheck=all
OPTIM = -O3 -march=native -ffast-math

# Use optimization flags (switch to DEBUG for debugging)
FFLAGS = $(OPTIM) -std=legacy -ffixed-line-length-none
# FFLAGS = $(DEBUG) -std=legacy -ffixed-line-length-none

# Directories
EXECU = ../exec/
OBDIR_ORIG = ../objets_orig/
OBDIR_NN = ../objets_nn/

# Source directories
REPERF_ORIG = ../fortran_original/
REPERF_NN = ../fortran_neural/

# Common object files (shared between both variants)
COMMON_OBJS = bgauss.o bunifo.o captur.o carltf.o carltz.o cartes.o \
	cisimu.o conphy.o energi.o enrtot.o entree.o ergols.o etafin.o etaini.o \
	faeros.o fatmos.o fgravi.o findal.o finmsr.o frayon.o fvents.o geodes.o \
	guialf.o guicap.o guidag.o guilat.o guilon.o inimsr.o intrmo.o intrde.o \
	interp.o lectci.o loteri.o matvec.o naviga.o opnfic.o orbito.o photra.o \
	pilote.o pnorme.o proalf.o pscalr.o pvecto.o realit.o reploc.o result.o \
	rkutta.o sequen.o simmsr.o sortie.o statis.o strlen.o tabmin.o tabref.o \
	tbgain.o vigite.o xvabsl.o

# Neural-network-only objects
NN_EXTRA_OBJS = guidnn.o lecgnn.o

# Full object lists with directory prefixes
OBJETS_ORIG = $(addprefix $(OBDIR_ORIG),$(COMMON_OBJS))
OBJETS_NN = $(addprefix $(OBDIR_NN),$(COMMON_OBJS) $(NN_EXTRA_OBJS))

# Targets
all: $(EXECU)aerocap $(EXECU)aerocap_nn

original: $(EXECU)aerocap
neural: $(EXECU)aerocap_nn

# Clean
clean:
	rm -f $(OBJETS_ORIG) $(OBJETS_NN) $(EXECU)aerocap $(EXECU)aerocap_nn

clean_orig:
	rm -f $(OBJETS_ORIG) $(EXECU)aerocap

clean_nn:
	rm -f $(OBJETS_NN) $(EXECU)aerocap_nn

# Link: original variant
$(EXECU)aerocap: $(OBJETS_ORIG)
	@[ -d $(EXECU) ] || mkdir -p $(EXECU)
	$(FC) $(FFLAGS) $(OBJETS_ORIG) -o $@

# Link: neural network variant
$(EXECU)aerocap_nn: $(OBJETS_NN)
	@[ -d $(EXECU) ] || mkdir -p $(EXECU)
	$(FC) $(FFLAGS) $(OBJETS_NN) -o $@

# Compile: original variant
$(OBDIR_ORIG)%.o: $(REPERF_ORIG)%.f
	@[ -d $(OBDIR_ORIG) ] || mkdir -p $(OBDIR_ORIG)
	$(FC) $(FFLAGS) -c $< -o $@

# Compile: neural network variant
$(OBDIR_NN)%.o: $(REPERF_NN)%.f
	@[ -d $(OBDIR_NN) ] || mkdir -p $(OBDIR_NN)
	$(FC) $(FFLAGS) -c $< -o $@

.PHONY: all original neural clean clean_orig clean_nn
