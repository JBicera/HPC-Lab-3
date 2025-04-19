#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "terarec.h"  // Provides terarec_t, teraCompare(), mpi_tera_type, and TERA_KEY_LEN

// Helper: stable counting sort by one key byte
static void countingSort(terarec_t *data, int len, int bytePos) {
    int count[256] = {0};
    terarec_t *output = malloc(len * sizeof(terarec_t));
    // Count occurrences
    for (int i = 0; i < len; i++) {
        unsigned char c = data[i].key[bytePos];
        count[c]++;
    }
    // Prefix sums
    for (int i = 1; i < 256; i++) count[i] += count[i-1];
    // Build output stable
    for (int i = len - 1; i >= 0; i--) {
        unsigned char c = data[i].key[bytePos];
        output[--count[c]] = data[i];
    }
    // Copy back
    memcpy(data, output, len * sizeof(terarec_t));
    free(output);
}

// Radix sort for fixed-length string key
static void radixSortRecords(terarec_t *data, int len) {
    // LSD radix: last byte to first
    for (int pos = TERA_KEY_LEN - 1; pos >= 0; pos--) {
        countingSort(data, len, pos);
    }
}

void terasort(terarec_t *local_data, int local_len,
              terarec_t **sortedData, int* sorted_counts, long* sorted_displs) {
    int rank, P;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Single large int buffer: sendCounts, recvCounts, sendDispl, recvDispl, bucketIdx, pos
    int numSamples = P - 1;
    int total_ints = 4*P + local_len + P;
    int *shared_ints = malloc(total_ints * sizeof(int));
    int *sendCounts = shared_ints;
    int *recvCounts = sendCounts + P;
    int *sendDispl  = recvCounts + P;
    int *recvDispl  = sendDispl  + P;
    int *bucketIdx  = recvDispl + P;
    int *pos        = bucketIdx  + local_len;

    // Buffers for data
    terarec_t *sendBuf      = malloc(local_len * sizeof(terarec_t));
    terarec_t *localSamples = malloc(numSamples * sizeof(terarec_t));
    terarec_t *splitters    = malloc(numSamples * sizeof(terarec_t));
    terarec_t *globalSamples = (rank==0)
        ? malloc(P * numSamples * sizeof(terarec_t))
        : NULL;

    // 1. Radix-sort local data
    radixSortRecords(local_data, local_len);

    // 2. Select P-1 samples
    for (int i = 0; i < numSamples; i++) {
        int idx = (i + 1) * local_len / P;
        if (idx >= local_len) idx = local_len - 1;
        localSamples[i] = local_data[idx];
    }

    // 3. Gather samples
    MPI_Gather(localSamples, numSamples, mpi_tera_type,
               globalSamples, numSamples, mpi_tera_type,
               0, MPI_COMM_WORLD);

    // 4. Radix-sort global samples & pick splitters on root
    if (rank == 0) {
        radixSortRecords(globalSamples, P * numSamples);
        for (int i = 1; i < P; i++)
            splitters[i-1] = globalSamples[i * numSamples];
    }
    // 5. Broadcast splitters
    MPI_Bcast(splitters, numSamples, mpi_tera_type, 0, MPI_COMM_WORLD);

    // 6. Partition into buckets
    memset(sendCounts, 0, P * sizeof(int));
    for (int i = 0; i < local_len; i++) {
        int b = 0;
        while (b < numSamples && teraCompare(&local_data[i], &splitters[b]) > 0) b++;
        bucketIdx[i] = b;
        sendCounts[b]++;
    }

    // 7. Compute send displacements & pack
    sendDispl[0] = 0;
    for (int i = 1; i < P; i++) sendDispl[i] = sendDispl[i-1] + sendCounts[i-1];
    memcpy(pos, sendDispl, P * sizeof(int));
    for (int i = 0; i < local_len; i++) {
        int b = bucketIdx[i];
        sendBuf[pos[b]++] = local_data[i];
    }

    // 8. Exchange counts
    MPI_Alltoall(sendCounts, 1, MPI_INT,
                 recvCounts, 1, MPI_INT,
                 MPI_COMM_WORLD);

    // 9. Compute recv displacements
    recvDispl[0] = 0;
    int totalRecv = recvCounts[0];
    for (int i = 1; i < P; i++) {
        recvDispl[i] = recvDispl[i-1] + recvCounts[i-1];
        totalRecv += recvCounts[i];
    }

    // 10. Exchange data
    terarec_t *recvBuf = malloc(totalRecv * sizeof(terarec_t));
    MPI_Alltoallv(sendBuf, sendCounts, sendDispl, mpi_tera_type,
                  recvBuf, recvCounts, recvDispl, mpi_tera_type,
                  MPI_COMM_WORLD);

    // 11. Final Radix-sort
    radixSortRecords(recvBuf, totalRecv);
    *sortedData = recvBuf;

    // 12. Gather final counts & compute global displs
    MPI_Allgather(&totalRecv, 1, MPI_INT,
                  sorted_counts, 1, MPI_INT,
                  MPI_COMM_WORLD);
    sorted_displs[0] = 0;
    for (int i = 1; i < P; i++)
        sorted_displs[i] = sorted_displs[i-1] + sorted_counts[i-1];

    // Cleanup
    free(shared_ints);
    free(sendBuf);
    free(localSamples);
    free(splitters);
    if (rank == 0) free(globalSamples);
}
