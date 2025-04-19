#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "terarec.h"  // Provides terarec_t, teraCompare(), mpi_tera_type, etc.

void terasort(terarec_t *local_data, int local_len,
              terarec_t **sortedData, int* sorted_counts, long* sorted_displs) {
    int rank, P;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Allocate a single int buffer to be sliced into counts, displs, bucketIdx, pos
    int total_ints = 4*P + local_len + P;
    int *shared_ints = malloc(total_ints * sizeof(int));
    int *sendCounts  = shared_ints;
    int *recvCounts  = shared_ints + P;
    int *sendDispl   = shared_ints + 2*P;
    int *recvDispl   = shared_ints + 3*P;
    int *bucketIdx   = shared_ints + 4*P;
    int *pos         = bucketIdx + local_len;  // last P entries

    // Allocate record buffers
    terarec_t *sendBuf      = malloc(local_len * sizeof(terarec_t));
    int numSamples = P - 1;
    terarec_t *localSamples = malloc(numSamples * sizeof(terarec_t));
    terarec_t *splitters    = malloc(numSamples * sizeof(terarec_t));
    terarec_t *globalSamples = (rank==0 ? malloc(P * numSamples * sizeof(terarec_t)) : NULL);

    // 1. Sort own data locally
    qsort(local_data, local_len, sizeof(terarec_t), teraCompare);

    // 2. Select P-1 samples from local_data
    for (int i = 0; i < numSamples; i++) {
        int idx = (i + 1) * local_len / P;
        if (idx >= local_len) idx = local_len - 1;
        localSamples[i] = local_data[idx];
    }

    // 3. Gather samples at root
    MPI_Gather(localSamples, numSamples, mpi_tera_type,
               globalSamples, numSamples, mpi_tera_type,
               0, MPI_COMM_WORLD);

    // 4. Sort global samples and pick splitters (root)
    if (rank == 0) {
        qsort(globalSamples, P * numSamples, sizeof(terarec_t), teraCompare);
        for (int i = 1; i < P; i++)
            splitters[i-1] = globalSamples[i * numSamples];
    }
    MPI_Bcast(splitters, numSamples, mpi_tera_type, 0, MPI_COMM_WORLD);

    // 5. Partition local_data into P buckets
    memset(sendCounts, 0, P * sizeof(int));
    for (int i = 0; i < local_len; i++) {
        int b = 0;
        while (b < numSamples && teraCompare(&local_data[i], &splitters[b]) > 0) b++;
        bucketIdx[i] = b;
        sendCounts[b]++;
    }

    // 6. Compute send displacements and pack sendBuf
    sendDispl[0] = 0;
    for (int i = 1; i < P; i++)
        sendDispl[i] = sendDispl[i-1] + sendCounts[i-1];
    memcpy(pos, sendDispl, P * sizeof(int));
    for (int i = 0; i < local_len; i++) {
        int b = bucketIdx[i];
        sendBuf[pos[b]++] = local_data[i];
    }

    // 7. Exchange bucket sizes
    MPI_Alltoall(sendCounts, 1, MPI_INT,
                 recvCounts, 1, MPI_INT,
                 MPI_COMM_WORLD);

    // 8. Compute recv displacements
    recvDispl[0] = 0;
    int totalRecv = recvCounts[0];
    for (int i = 1; i < P; i++) {
        recvDispl[i] = recvDispl[i-1] + recvCounts[i-1];
        totalRecv += recvCounts[i];
    }

    // 9. Exchange data
    terarec_t *recvBuf = malloc(totalRecv * sizeof(terarec_t));
    MPI_Alltoallv(sendBuf, sendCounts, sendDispl, mpi_tera_type,
                  recvBuf, recvCounts, recvDispl, mpi_tera_type,
                  MPI_COMM_WORLD);

    // 10. Final local sort
    qsort(recvBuf, totalRecv, sizeof(terarec_t), teraCompare);
    *sortedData = recvBuf;

    // 11. Gather final counts and compute global displacements
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
