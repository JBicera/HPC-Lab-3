#ifndef __TERAREC_H__
#define __TERAREC_H__

#include <mpi.h>

#define TERA_KEY_LEN 10
#define TERA_VALUE_LEN 88

typedef struct{
	char key[TERA_KEY_LEN + 1];
	char value[TERA_VALUE_LEN + 1];
} terarec_t;

extern MPI_Datatype mpi_tera_type;

//Registers the terarec_t type with MPI
void teraMPICommitType();

//A comparison function for the terarec_t type returning
// < 0 if a->key < b->key
// 0 if a->key = b->key
// > 0 if a->key > b->key
int teraCompare(const void *a, const void *b);


//Randomly generates len terarec values understood
//to start of offset within some larger sequence
terarec_t* teraGenerate(long offset, int len);


//Returns the number of terarec stored in a file
long teraRecsInFile(int fd);

//Stores in the returned array an equal share (among
//all processies in COMM_WORLD) of the data in filename.
//The number of records in returned in the return value len.
terarec_t* readEqualShare(char* filename, int *len);


//Reads nrecs from the the file descriptor fd staring at offset records 
//into the file. The result is stored in the array recs. 
int teraReadFromFile(int fd, long offset, int nrecs, terarec_t *recs);

//Writes nrecs from the array recs to the file descriptor fd 
//offset records into the file.
int teraWriteToFile(int fd, long offset, int nrecs, terarec_t *recs);

#endif
