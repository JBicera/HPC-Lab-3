#define _XOPEN_SOURCE 500

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stddef.h>

#include "terarec.h"

#define TERA_REC_WIDTH 100
#define RECS_IN_BUFFER 40000

MPI_Datatype mpi_tera_type;

//Adapted from Apache TeraSort
static void assignKey(char *key){

	for(int i = 0; i < 3; i++){
	    int  temp = rand() / 52;
	    if( i < 2)
		    key[3 + 4*i] = (char) (' ' + (temp % 95));
        temp /= 95;
		if (i < 2)
	        key[2 + 4*i] = (char) (' ' + (temp % 95));
        temp /= 95;
        key[1 + 4*i] = (char) (' ' + (temp % 95));
		temp /= 95;
		key[4*i] = (char) (' ' + (temp % 95));
    }
    key[TERA_KEY_LEN] = '\0';
}

static void assignRowId(char *value, long rowid){
	sprintf(value, "%010ld", rowid);
}

static void assignFiller(char *value, long rowid){
    long base = (long) ((rowid * 8) % 26);
	for(int i=0; i<8; ++i) {
		char c = 'A' + (base + i) % 26;
		for(int j = 0; j < 10 && (10 * (i+1) + j) < (TERA_VALUE_LEN); j++)
			value[10 * (i+1) + j] = c;
	}
    value[TERA_VALUE_LEN] = '\0';
}

void teraMPICommitType(){
	static int committed = 0;

	if(committed)
		return;

    int blocklengths[2] = {TERA_KEY_LEN+1, TERA_VALUE_LEN+1};
    MPI_Datatype types[2] = {MPI_CHAR, MPI_CHAR};

    MPI_Aint     offsets[2];
    offsets[0] = offsetof(terarec_t, key);
    offsets[1] = offsetof(terarec_t, value);

    MPI_Type_create_struct(2, blocklengths, offsets, types, &mpi_tera_type);
    MPI_Type_commit(&mpi_tera_type);

    committed = 1;
}

terarec_t* teraGenerate(long offset, int  len){
	terarec_t* ans = (terarec_t *) malloc(len * sizeof(terarec_t));

	if (!ans)
		return ans;

	for(int  i = 0; i < len; i++){
		assignKey(&ans[i].key[0]);
		assignRowId(&ans[i].value[0], offset + i);
		assignFiller(&ans[i].value[0], offset + i);
	}

	return ans;
}

long teraRecsInFile(int fd){
	size_t file_length = lseek(fd, 0, SEEK_END);
	return file_length / TERA_REC_WIDTH;
}

terarec_t* readEqualShare(char* filename, int *_len){
	int P, rank;
	MPI_Comm_size (MPI_COMM_WORLD, &P);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	int fd;
	if(0 > (fd = open(filename, O_RDONLY))){
		fprintf(stderr, "Failed to open file %s.", filename);
		exit(EXIT_FAILURE);
	}

	int nrecs = teraRecsInFile(fd);
	int len = nrecs / P + (rank < (nrecs % P) ? 1 : 0);
	int file_offset = rank * len + (rank < (nrecs % P) ? 0 : (nrecs % P));

	terarec_t *local_data = (terarec_t*) malloc(len * sizeof(terarec_t));

	teraReadFromFile(fd, file_offset, len, local_data);

	close(fd);

	*_len = len;

	return local_data;
}


int teraReadFromFile(int fd, long offset, int  nrecs, terarec_t *recs){
	char buffer[RECS_IN_BUFFER * TERA_REC_WIDTH];
	int  nread = 0;

	while(nread < nrecs){
		int  goal_recs = (nrecs - nread < RECS_IN_BUFFER) ? (nrecs - nread) : RECS_IN_BUFFER;
		int  goal_read = TERA_REC_WIDTH * goal_recs;
		int  bytes_read = 0;

		while(bytes_read < goal_read){
			int retval = pread(fd, &buffer[bytes_read], goal_read - bytes_read, (offset + nread) * TERA_REC_WIDTH + bytes_read);
			if(retval == -1){
				fprintf(stderr, "pwrite error in teraWriteToFile : %s\n", strerror(errno));
				return EXIT_FAILURE;
			}
			bytes_read += retval;
		}

		for(int  i = 0; i < goal_recs; i++){
			memcpy(&recs[nread+i].key[0], &buffer[i * TERA_REC_WIDTH], TERA_KEY_LEN);
			recs[nread+i].key[TERA_KEY_LEN] = '\0';

			memcpy(&recs[nread+i].value[0], &buffer[i * TERA_REC_WIDTH + TERA_KEY_LEN],  TERA_VALUE_LEN);
			recs[nread+i].value[TERA_VALUE_LEN] = '\0';
		}

		nread += goal_recs;
	}

	return EXIT_SUCCESS;

}

int teraWriteToFile(int fd, long offset, int  nrecs, terarec_t *recs){
	char buffer[RECS_IN_BUFFER * TERA_REC_WIDTH];
	int  nwritten = 0;

	while(nwritten < nrecs){
		int  goal_recs = (nrecs - nwritten < RECS_IN_BUFFER) ? (nrecs - nwritten) : RECS_IN_BUFFER;
		int  goal_written = TERA_REC_WIDTH * goal_recs;
		int  bytes_written = 0;

		for(int  i = 0; i < goal_recs; i++){
			memcpy(&buffer[i * TERA_REC_WIDTH], &recs[nwritten+i].key[0], TERA_KEY_LEN);
			memcpy(&buffer[i * TERA_REC_WIDTH + TERA_KEY_LEN], &recs[nwritten+i].value[0], TERA_VALUE_LEN);
		
			buffer[(i+1) * TERA_REC_WIDTH - 2] = '\r';
			buffer[(i+1) * TERA_REC_WIDTH - 1] = '\n';
		}

		while(bytes_written < goal_written){
			int retval = pwrite(fd, &buffer[bytes_written], goal_written - bytes_written, (offset + nwritten) * TERA_REC_WIDTH + bytes_written);
			if(retval == -1){
				fprintf(stderr, "pwrite error in teraWriteToFile : %s\n", strerror(errno));
				return EXIT_FAILURE;
			}
			bytes_written += retval;
		}
		nwritten += goal_recs;
	}

	return EXIT_SUCCESS;
}

int teraCompare(const void *a, const void *b){
	return memcmp( ((terarec_t*) a)->key, ((terarec_t*) b)->key, TERA_KEY_LEN);
}
