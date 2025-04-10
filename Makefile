MPICC = mpicc
MPICFLAGS = -std=c99 -fopenmp
MPICOPTFLAGS = -Iutils -O0 -g -Wall -Werror
MPILDFLAGS =

teragen: utils/teragen_main.c utils/terarec.c
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS) -o $@ $^ $(MPILDFLAGS)

terasort: utils/terasort_main.c student/terasort.c utils/terarec.c
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS) -o $@ $^ $(MPILDFLAGS)

teravalidate: utils/teravalidate_main.c utils/teravalidate.c utils/terarec.c
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS) -o $@ $^ $(MPILDFLAGS)
	
terametrics: utils/terametrics.c student/terasort.c utils/terarec.c utils/teravalidate.c
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS) -o $@ $^ $(MPILDFLAGS)

naivesort: utils/naivesort.c utils/terarec.c
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS) -o $@ $^ $(MPILDFLAGS)

.PHONY: clean

clean:
	rm -f *.o teragen terasort teravalidate terametrics naivesort data.dat
