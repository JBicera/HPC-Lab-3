#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "terarec.h"  // Provides terarec_t, teraCompare(), mpi_tera_type, etc.

// Helper function for Radix Sort: counting sort for a specific digit position
void countingSort(terarec_t *data, int len, int exp) 
{
    terarec_t *output = malloc(len * sizeof(terarec_t));
    int count[256] = {0};  // For 256 ASCII characters

    // Store count of occurrences in count[]
    for (int i = 0; i < len; i++) {
        unsigned char digit = data[i].key[exp];
        count[digit]++;
    }

    // Change count[i] so that it now contains the actual position of this digit in output[]
    for (int i = 1; i < 256; i++) {
        count[i] += count[i - 1];
    }

    // Build the output array using the count[] to place the correct elements in the output array
    for (int i = len - 1; i >= 0; i--) {
        unsigned char digit = data[i].key[exp];
        output[count[digit] - 1] = data[i];
        count[digit]--;
    }

    // Copy the output array to data[], so that data[] now contains sorted numbers
    memcpy(data, output, len * sizeof(terarec_t));
    free(output);
}

// Radix Sort function: sorts the records based on their key
void radixSort(terarec_t *data, int len) 
{
    // Find the maximum number in the data to know the number of digits
    int max = 0;
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < TERA_KEY_LEN; j++) {
            if (data[i].key[j] > max) {
                max = data[i].key[j];
            }
        }
    }

    // Apply counting sort for every digit. The exp is 1, 10, 100, etc.
    for (int exp = 0; exp < TERA_KEY_LEN; exp++) {
        countingSort(data, len, exp);
    }
}

void terasort(terarec_t *local_data, int local_len, 
              terarec_t **sortedData, int* sorted_counts, long* sorted_displs) {
    int rank, P;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 1. Sort own data locally using radix sort (instead of quicksort)
    radixSort(local_data, local_len);

    // 2. Select P-1 samples from local sorted data
    int numSamples = P - 1;
    terarec_t *localSamples = malloc(numSamples * sizeof(terarec_t)); // Store samples
    for (int i = 0; i < numSamples; i++) {
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
    if (rank == 0) {
        radixSort(globalSamples, numSamples * P); // Sort samples
        for (int i = 1; i < P; i++) {
            splitters[i - 1] = globalSamples[i * numSamples]; // Pick every Pth element from samples
        }
        free(globalSamples);
    }

    // 5. Broadcast splitters
    MPI_Bcast(splitters, P - 1, mpi_tera_type, 0, MPI_COMM_WORLD); // Send splitters to every process

    // 6. Partition local data using splitters
    int *sendCount = calloc(P, sizeof(int));  // Count of elements to send to each process
    int *bucketIndices = malloc(local_len * sizeof(int)); // Keep track of bucket indices for each record

    // Count how many items go in each bucket using teraCompare
    for (int i = 0; i < local_len; i++) {
        int bucket = 0;
        while (bucket < P - 1 && teraCompare(&local_data[i], &splitters[bucket]) > 0)
            bucket++;
        bucketIndices[i] = bucket;
        sendCount[bucket]++;
    }
    free(splitters);

    // Compute send displacements
    int *sendDispl = malloc(P * sizeof(int));
    sendDispl[0] = 0;
    for (int i = 1; i < P; i++)
        sendDispl[i] = sendDispl[i - 1] + sendCount[i - 1];

    // Allocate send buffer and temp position tracker
    terarec_t *sendBuff = malloc(local_len * sizeof(terarec_t));
    int *tempCounts = calloc(P, sizeof(int));

    // Copy data to send buffer
    for (int i = 0; i < local_len; i++) {
        int bucket = bucketIndices[i];
        int pos = sendDispl[bucket] + tempCounts[bucket]++; // Allocate to proper bucket
        sendBuff[pos] = local_data[i];
    }
    free(tempCounts);

    // 7. Exchange partitioned data
    int *recvCounts = malloc(P * sizeof(int));
    MPI_Alltoall(sendCount, 1, MPI_INT, recvCounts, 1, MPI_INT, MPI_COMM_WORLD);

    int *recvDispl = malloc(P * sizeof(int));
    recvDispl[0] = 0;
    int totalRecv = recvCounts[0];
    for (int i = 1; i < P; i++) {
        recvDispl[i] = recvDispl[i - 1] + recvCounts[i - 1];
        totalRecv += recvCounts[i];
    }

    terarec_t *recvBuff = malloc(totalRecv * sizeof(terarec_t));
    MPI_Alltoallv(sendBuff, sendCount, sendDispl, mpi_tera_type,
                  recvBuff, recvCounts, recvDispl, mpi_tera_type,
                  MPI_COMM_WORLD);

    free(sendBuff);
    free(sendCount);
    free(sendDispl);
    free(recvDispl);

    // 8. Final local sort using teraCompare
    qsort(recvBuff, totalRecv, sizeof(terarec_t), teraCompare); // Use qsort and teraCompare for final sort
    *sortedData = recvBuff; // Return values
    int count = totalRecv;
    MPI_Allgather(&count, 1, MPI_INT, sorted_counts, 1, MPI_INT, MPI_COMM_WORLD);

    // Compute displacements for the global sorted array
    sorted_displs[0] = 0;
    for (int i = 1; i < P; i++)
        sorted_displs[i] = sorted_displs[i - 1] + sorted_counts[i - 1];
}
