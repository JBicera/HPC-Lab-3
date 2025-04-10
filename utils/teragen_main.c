#define _OPEN_SOURCE 500

#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <getopt.h>
#include <mpi.h>
#include "terarec.h"


static long seeds[32] = {0L, 
				4160749568L, 
				4026531840L, 
				3892314112L,
				3758096384L,
				3623878656L,
				3489660928L,
				3355443200L,
				3221225472L,
				3087007744L,
				2952790016L,
				2818572288L,
				2684354560L,
				2550136832L,
				2415919104L,
				2281701376L,
				2147483648L,
				2013265920L,
				1879048192L,
				1744830464L,
				1610612736L,
				1476395008L,
				1342177280L,
				1207959552L,
				1073741824L,
				939524096L,
				805306368L,
				671088640L,
				536870912L,
				402653184L,
				268435456L,
				134217728L};

#define USAGE \
"usage:\n"                                                                     \
"  teragen [options]\n"                                                        \
"options:\n"                                                                   \
"  -f [filename]       Filename (Default: data.dat)\n"                         \
"  -c [count]    	   Elements to generate per processor (Default: 400)\n"\
"  -h                  Show this help message\n"

extern int zerocount;

int main (int argc, char *argv[]){
	int P, rank;
	MPI_Init (&argc, &argv);
	MPI_Comm_size (MPI_COMM_WORLD, &P);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	char * filename = "data.dat"; 
	long nrecs = 400;
	int c;
	int safeWriteCount;

	while ((c = getopt (argc, argv, "hf:c:")) != -1){
		switch (c){
			case 'f':
				filename = optarg;
				break;
			case 'c':
				nrecs = atol(optarg);
				break;
			case 'h':
				fprintf(stderr, USAGE);
				exit(EXIT_SUCCESS);
			default:
				fprintf(stderr, USAGE);
				exit(EXIT_FAILURE);
		}
	}

	//Generating Data
	srand(seeds[rank % 32]);
	terarec_t *data = teraGenerate(rank * nrecs, nrecs);

	int fd;
	if(0 > (fd = open(filename, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR))){
		fprintf(stderr, "Failed to open file %s.", filename);
		exit(EXIT_FAILURE);
	}

	for (safeWriteCount=0; safeWriteCount < rank; safeWriteCount++)
		sleep(1);

	teraWriteToFile(fd, rank * nrecs, nrecs, data);

	close(fd);

	MPI_Finalize();

	return EXIT_SUCCESS;
}
