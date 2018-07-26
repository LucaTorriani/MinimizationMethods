DIFF ?= diff
MPIEXEC ?= mpiexec
CXX = mpicxx

EXE = main_rosenbrock

CXXFLAGS += -Wall -std=c++11 -pedantic -Wextra
MPIFLAGS += -np 1 $(EXE) 2> /dev/null

OBJS = Monomial.o Point.o Dense_Matrix.o Linear_System.o FunctionRn.o FunctionRn_Constrained.o MPI_helpers.o $(EXE).o

Point.o: Point.hh
Monomial.o:  Monomial.hh Point.hh
FunctionRn.o: FunctionRn.hh Monomial.hh Point.hh Dense_Matrix.hh
FunctionRn_Constrained.o: FunctionRn_Constrained.hh FunctionRn.hh Point.hh Dense_Matrix.hh
Dense_Matrix.o: Dense_Matrix.hh Point.hh
Linear_System.o: Linear_System.hh Dense_Matrix.hh Point.hh
MPI_helpers.o: MPI_helpers.hh
$(EXE).o: Point.hh Monomial.hh FunctionRn.hh FunctionRn_Constrained.hh Dense_Matrix.hh MPI_helpers.hh GradientDescent.hh Newton.hh QuasiNewton.hh Constrained_Min.hh Unconstrained_Min.hh Backtrack.hh

.DEFAULT_GOAL = all
all: build

.PHONY: all build clean distclean run

build: $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) $^ $(OUTPUT_OPTION)

run: $(EXE)
	$(MPIEXEC) $(MPIFLAGS) ./$(EXE)

clean:
	$(RM) *.o
	$(RM) $(EXE)

distclean: clean
	$(RM) *.~
