#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "terarec.h"  // Provides terarec_t, teraCompare(), mpi_tera_type, etc.

#define BYTE_AT(key, d) ((unsigned char)(key[d]))
#define MAX_DIGIT 10  // up to 10 bytes of key for MSD radix sort

static void msdRadixSortRecords(terarec_t *A, int n, int digit) {
    if (n <= 1 || digit >= MAX_DIGIT) return;

    int count[256] = {0};
    int index[256];

    for (int i = 0; i < n; i++) {
        unsigned char d = BYTE_AT(A[i].key, digit);
        count[d]++;
    }

    index[0] = 0;
    for (int i = 1; i < 256; i++) {
        index[i] = index[i - 1] + count[i - 1];
    }

    terarec_t *temp = malloc(n * sizeof(terarec_t));
    for (int i = 0; i < n; i++) {
        unsigned char d = BYTE_AT(A[i].key, digit);
        temp[index[d]++] = A[i];
    }
    memcpy(A, temp, n * sizeof(terarec_t));
    free(temp);

    index[0] = 0;
    for (int i = 1; i < 256; i++) {
        index[i] = index[i - 1] + count[i - 1];
    }
    for (int i = 0; i < 256; i++) {
        if (count[i] > 1) {
            msdRadixSortRecords(A + index[i], count[i], digit + 1);
        }
    }
}

void terasort(terarec_t *local_data, int local_len,
              terarec_t **sortedData, int* sorted_counts, long* sorted_displs) {
    int rank, P;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Buffers reused across phases
    int *sendCounts  = calloc(P, sizeof(int));
    int *sendDispl   = malloc(P * sizeof(int));
    int *recvCounts  = malloc(P * sizeof(int));
    int *recvDispl   = malloc(P * sizeof(int));
    terarec_t *sendBuf = malloc(local_len * sizeof(terarec_t));

    // 1. Sort own data locally (MSD Radix)
    msdRadixSortRecords(local_data, local_len, 0);

    // 2. Select P-1 samples
    int numSamples = P - 1;
    terarec_t *localSamples = malloc(numSamples * sizeof(terarec_t));
    for (int i = 0; i < numSamples; i++) {
        int idx = (i + 1) * local_len / P;
        if (idx >= local_len) idx = local_len - 1;
        localSamples[i] = local_data[idx];
    }

    // 3. Gather samples
    terarec_t *globalSamples = NULL;
    if (rank == 0)
        globalSamples = malloc(P * numSamples * sizeof(terarec_t));
    MPI_Gather(localSamples, numSamples, mpi_tera_type,
               globalSamples, numSamples, mpi_tera_type,
               0, MPI_COMM_WORLD);
    free(localSamples);

    // 4. Sort samples on root (MSD Radix)
    terarec_t *splitters = malloc(numSamples * sizeof(terarec_t));
    if (rank == 0) 
    {
        msdRadixSortRecords(globalSamples, P * numSamples, 0);

        // 5. Select P - 1 splitters
        for (int i = 1; i < P; i++)
            splitters[i-1] = globalSamples[i * numSamples];
        free(globalSamples);
    }

    // 6. Broadcast splitters
    MPI_Bcast(splitters, numSamples, mpi_tera_type, 0, MPI_COMM_WORLD);

    // 7. Partition using splitters
    memset(sendCounts, 0, P * sizeof(int));
    int *bucketIdx = malloc(local_len * sizeof(int));
    for (int i = 0; i < local_len; i++) {
        int b = 0;
        while (b < P-1 &&
               teraCompare(&local_data[i], &splitters[b]) > 0) b++;
        bucketIdx[i] = b;
        sendCounts[b]++;
    }
    free(splitters);

    // Compute send displacements
    sendDispl[0] = 0;
    for (int i = 1; i < P; i++)
        sendDispl[i] = sendDispl[i-1] + sendCounts[i-1];

    int *pos = malloc(P * sizeof(int));
    memcpy(pos, sendDispl, P * sizeof(int));

    // Pack buffer
    for (int i = 0; i < local_len; i++)
        sendBuf[pos[bucketIdx[i]]++] = local_data[i];
    free(pos);
    free(bucketIdx);

    // 8. Exchange counts 
    MPI_Alltoall(sendCounts, 1, MPI_INT, recvCounts, 1, MPI_INT, MPI_COMM_WORLD);

    // Compute recv displacements
    recvDispl[0] = 0;
    int totalRecv = recvCounts[0];
    for (int i = 1; i < P; i++) 
    {
        recvDispl[i] = recvDispl[i-1] + recvCounts[i-1];
        totalRecv += recvCounts[i];
    }

    // Alltoallv data exchange
    terarec_t *recvBuf = malloc(totalRecv * sizeof(terarec_t));
    MPI_Alltoallv(sendBuf, sendCounts, sendDispl, mpi_tera_type,
                  recvBuf, recvCounts, recvDispl, mpi_tera_type,
                  MPI_COMM_WORLD);
    free(sendBuf);
    free(sendCounts);
    free(sendDispl);

    // 9. Final local sort (MSD Radix)
    msdRadixSortRecords(recvBuf, totalRecv, 0);
    *sortedData = recvBuf;

    // Gather final counts and compute displs
    MPI_Allgather(&totalRecv, 1, MPI_INT,
                  sorted_counts, 1, MPI_INT,
                  MPI_COMM_WORLD);
    sorted_displs[0] = 0;
    for (int i = 1; i < P; i++)
        sorted_displs[i] = sorted_displs[i-1] + sorted_counts[i-1];

    free(recvCounts);
    free(recvDispl);
}