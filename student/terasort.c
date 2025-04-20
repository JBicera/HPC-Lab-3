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

    // Allocate integer buffers
    int *sendCounts = calloc(P, sizeof(int));    // bucket sizes for send
    int *sendDispl  = malloc(P * sizeof(int));   // send offsets
    int *recvCounts = malloc(P * sizeof(int));   // bucket sizes for recv
    int *recvDispl  = malloc(P * sizeof(int));   // recv offsets

    // Buffer for packing outgoing records
    terarec_t *sendBuf = malloc(local_len * sizeof(terarec_t));

    // 1. Local sort the local data
    qsort(local_data, local_len, sizeof(terarec_t), teraCompare);

    // 2. Select P - 1 Samples
    int numSamples = P - 1;
    terarec_t *samples = malloc(P * numSamples * sizeof(terarec_t));
    // Evenly spaced samples from local sorted data
    for (int i = 0; i < numSamples; i++) 
    {
        int idx = (i + 1) * local_len / P;
        if (idx >= local_len) 
            idx = local_len - 1;
        samples[i] = local_data[idx];
    }

    // 3. Gather samples to root
    MPI_Gather(samples, numSamples, mpi_tera_type,
               samples, numSamples, mpi_tera_type,
               0, MPI_COMM_WORLD);

    // 4. Sort samples on root 
    if (rank == 0) 
    {
        // Sort using quick sort
        qsort(samples, P * numSamples, sizeof(terarec_t), teraCompare);
        // 5. Select P - 1 Splitters
        for (int i = 1; i < P; i++)
            samples[i - 1] = samples[i * numSamples];
    }

    // 6. Broadcast splitters
    MPI_Bcast(samples, numSamples, mpi_tera_type, 0, MPI_COMM_WORLD);

    // 7. Partition using splitters
    memset(sendCounts, 0, P * sizeof(int));
    int *bucketIdx = malloc(local_len * sizeof(int)); // Store bucekt index
    for (int i = 0; i < local_len; i++) 
    {
        int bucket = 0;
        while (bucket < numSamples && teraCompare(&local_data[i], &samples[bucket]) > 0)
            bucket++;
        bucketIdx[i] = bucket;
        sendCounts[bucket]++;
    }

    // Compute send displacements
    sendDispl[0] = 0;
    for (int i = 1; i < P; i++)
        sendDispl[i] = sendDispl[i - 1] + sendCounts[i - 1]; // Computer where each bucket should start

    // Pack sendBuf according to buckets
    int *pos = malloc(P * sizeof(int));
    memcpy(pos, sendDispl, P * sizeof(int));
    for (int i = 0; i < local_len; i++) 
        sendBuf[pos[bucketIdx[i]]++] = local_data[i];

    free(bucketIdx);
    free(pos);

    // Exchange counts
    MPI_Alltoall(sendCounts, 1, MPI_INT, recvCounts, 1, MPI_INT,MPI_COMM_WORLD);

    // Compute recv displacements
    recvDispl[0] = 0;
    int totalRecv = recvCounts[0];
    for (int i = 1; i < P; i++) 
    {
        recvDispl[i] = recvDispl[i - 1] + recvCounts[i - 1]; // Where incoming data from other ranks should be inside recvbuf
        totalRecv += recvCounts[i];
    }

    // 8. Exchange values
    terarec_t *recvBuf = malloc(totalRecv * sizeof(terarec_t));
    MPI_Alltoallv(sendBuf, sendCounts, sendDispl, mpi_tera_type,
                  recvBuf, recvCounts, recvDispl, mpi_tera_type,
                  MPI_COMM_WORLD);

    // Cleanup send buffers
    free(sendBuf);
    free(sendCounts);
    free(sendDispl);

    // 9. Final local sort
    qsort(recvBuf, totalRecv, sizeof(terarec_t), teraCompare);
    *sortedData = recvBuf;

    // Gather final counts and compute global displacements
    MPI_Allgather(&totalRecv, 1, MPI_INT,
                  sorted_counts, 1, MPI_INT,
                  MPI_COMM_WORLD);
    sorted_displs[0] = 0; 
    for (int i = 1; i < P; i++)
        sorted_displs[i] = sorted_displs[i - 1] + sorted_counts[i - 1]; // Where each rank's data begins globally

    free(recvCounts);
    free(recvDispl);
    free(samples);
}
