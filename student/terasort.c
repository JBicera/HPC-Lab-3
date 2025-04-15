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

    // 1. Sort own data locally using quicksort
    qsort(local_data, local_len, sizeof(terarec_t), teraCompare);

    // 2. Select P-1 samples from local sorted data
    int numSamples = P - 1;
    terarec_t *localSamples = malloc(numSamples * sizeof(terarec_t)); // Store samples
    for (int i = 0; i < numSamples; i++) 
    {
        int index = (i + 1) * local_len / P;
        if (index >= local_len) 
            index = local_len - 1;
        localSamples[i] = local_data[index];
    }

    // 3. Gather samples at root
    terarec_t *globalSamples = NULL;
    if (rank == 0)
        globalSamples = malloc(P * numSamples * sizeof(terarec_t));
    MPI_Gather(localSamples, numSamples, mpi_tera_type,
               globalSamples, numSamples, mpi_tera_type,
               0, MPI_COMM_WORLD); // MPI_Gather to root
    free(localSamples);

    // 4. Root sorts samples and selects P-1 splitters
    terarec_t *splitters = malloc((P - 1) * sizeof(terarec_t));
    if (rank == 0) 
    {
        qsort(globalSamples, numSamples * P, sizeof(terarec_t), teraCompare);
        for (int i = 1; i < P; i++)
            splitters[i - 1] = globalSamples[i * numSamples]; // Pick every Pth element form samples
        free(globalSamples);
    }

    // 5. Broadcast splitters
    MPI_Bcast(splitters, P - 1, mpi_tera_type, 0, MPI_COMM_WORLD); // Send splitters to every process

    // 6. Partition local data using splitters
    int *sendCount = calloc(P, sizeof(int));  // Count of elements to send to each process

    // Count how many items go in each bucket
    for (int i = 0; i < local_len; i++) 
    {
        int bucket = 0;
        while (bucket < P - 1 && teraCompare(&local_data[i], &splitters[bucket]) > 0)
            bucket++;
        sendCount[bucket]++;
    }

    // Compute send displacements
    int *sendDispl = malloc(P * sizeof(int));
    sendDispl[0] = 0;
    for (int i = 1; i < P; i++)
        sendDispl[i] = sendDispl[i - 1] + sendCount[i - 1];

    // Allocate send buffer and temp position tracker
    terarec_t *sendBuff = malloc(local_len * sizeof(terarec_t));
    int *tempCounts = calloc(P, sizeof(int));

    // Copy data to send buffer
    for (int i = 0; i < local_len; i++) 
    {
        int bucket = 0;
        while (bucket < P - 1 && teraCompare(&local_data[i], &splitters[bucket]) > 0)
            bucket++;
        int pos = sendDispl[bucket] + tempCounts[bucket]++;
        sendBuff[pos] = local_data[i];
    }

    free(tempCounts);
    free(splitters);

    // 7. Exchange partitioned data
    int *recvCounts = malloc(P * sizeof(int));
    MPI_Alltoall(sendCount, 1, MPI_INT, recvCounts, 1, MPI_INT, MPI_COMM_WORLD);

    int *recvDispl = malloc(P * sizeof(int));
    recvDispl[0] = 0;
    int total_recv = recvCounts[0];
    for (int i = 1; i < P; i++) 
    {
        recvDispl[i] = recvDispl[i - 1] + recvCounts[i - 1];
        total_recv += recvCounts[i];
    }

    terarec_t *recvBuff = malloc(total_recv * sizeof(terarec_t));
    MPI_Alltoallv(sendBuff, sendCount, sendDispl, mpi_tera_type,
                  recvBuff, recvCounts, recvDispl, mpi_tera_type,
                  MPI_COMM_WORLD);

    free(sendBuff);
    free(sendCount);
    free(sendDispl);
    free(recvDispl);

    // 8. Final local sort
    qsort(recvBuff, total_recv, sizeof(terarec_t), teraCompare);

    *sortedData = recvBuff; // Return values

    int my_count = total_recv;
    MPI_Allgather(&my_count, 1, MPI_INT, sorted_counts, 1, MPI_INT, MPI_COMM_WORLD);

    sorted_displs[0] = 0;
    for (int i = 1; i < P; i++) {
        sorted_displs[i] = sorted_displs[i - 1] + sorted_counts[i - 1];
    }
}
