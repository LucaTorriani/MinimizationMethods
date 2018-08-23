DIFF ?= diff
MPIEXEC ?= mpiexec
CXX = mpicxx

EXE = main_7_15 main_7_18 main_beale main_h main_rosenbrock main_rosenbrock2

CXXFLAGS += -Wall -std=c++11 -pedantic -Wextra
MPIFLAGS += -np 1

OBJS = Monomial.o Point.o Dense_Matrix.o Linear_System.o FunctionRn.o FunctionRn_Constrained.o MPI_helpers.o
OBJS_MAIN = main_7_15.o main_7_18.o main_beale main_h main_rosenbrock main_rosenbrock2
HEADERS_CONSTR = Point.hh Monomial.hh FunctionRn.hh FunctionRn_Constrained.hh Dense_Matrix.hh MPI_helpers.hh GradientDescent.hh Newton.hh QuasiNewton.hh Constrained_Min.hh Backtrack.hh
HEADERS_UNCONSTR = Point.hh Monomial.hh FunctionRn.hh Dense_Matrix.hh MPI_helpers.hh GradientDescent.hh Newton.hh QuasiNewton.hh Unconstrained_Min.hh Backtrack.hh

Point.o: Point.hh
Monomial.o: Monomial.hh Point.hh
FunctionRn.o: FunctionRn.hh Monomial.hh Point.hh Dense_Matrix.hh
FunctionRn_Constrained.o: FunctionRn_Constrained.hh FunctionRn.hh Point.hh Dense_Matrix.hh
Dense_Matrix.o: Dense_Matrix.hh Point.hh
Linear_System.o: Linear_System.hh Dense_Matrix.hh Point.hh
MPI_helpers.o: MPI_helpers.hh
main_7_15.o: $(HEADERS_CONSTR)
main_7_18.o: $(HEADERS_CONSTR)
main_beale.o: $(HEADERS_UNCONSTR)
main_h.o: $(HEADERS_CONSTR)
main_rosenbrock.o: $(HEADERS_CONSTR)
main_rosenbrock2.o: $(HEADERS_CONSTR)

.DEFAULT_GOAL = all
all: build

.PHONY: all build clean distclean run

build: $(EXE)

main_7_15: $(OBJS) main_7_15.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) $^ $(OUTPUT_OPTION)

main_7_18: $(OBJS) main_7_18.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) $^ $(OUTPUT_OPTION)

main_beale: $(OBJS) main_beale.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) $^ $(OUTPUT_OPTION)

main_h: $(OBJS) main_h.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) $^ $(OUTPUT_OPTION)

main_rosenbrock: $(OBJS) main_rosenbrock.o
		$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) $^ $(OUTPUT_OPTION)

main_rosenbrock2: $(OBJS) main_rosenbrock2.o
			$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) $^ $(OUTPUT_OPTION)

run: $(ARGS)
	$(MPIEXEC) $(MPIFLAGS) ./$(ARGS)

clean:
	$(RM) *.o
	$(RM) $(EXE)

distclean: clean
	$(RM) *.~
