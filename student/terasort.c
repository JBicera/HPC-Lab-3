#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "terarec.h"  // Provides terarec_t, teraCompare(), mpi_tera_type, etc.

void terasort(terarec_t *local_data, int local_len, 
              terarec_t **sorted_data, int* sorted_counts, long* sorted_displs) {
    int rank, P;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 1. Sort own data locally
    qsort(local_data, local_len, sizeof(terarec_t), teraCompare);

    // 2. Select P-1 samples from local sorted data
    int nsamples = P - 1;
    terarec_t *localSamples = malloc(nsamples * sizeof(terarec_t));
    for (int i = 0; i < nsamples; i++) {
        int index = (i + 1) * local_len / P;
        if (index >= local_len) index = local_len - 1;
        localSamples[i] = local_data[index];
    }

    // 3. Gather samples at root
    terarec_t *globalSamples = NULL;
    if (rank == 0) {
        globalSamples = malloc(P * nsamples * sizeof(terarec_t));
    }
    MPI_Gather(localSamples, nsamples, mpi_tera_type,
               globalSamples, nsamples, mpi_tera_type,
               0, MPI_COMM_WORLD);
    free(localSamples);

    // 4. Root sorts samples and selects P-1 splitters
    terarec_t *splitters = malloc((P - 1) * sizeof(terarec_t));
    if (rank == 0) {
        qsort(globalSamples, nsamples * P, sizeof(terarec_t), teraCompare);
        for (int i = 1; i < P; i++) {
            splitters[i - 1] = globalSamples[i * nsamples];
        }
        free(globalSamples);
    }

    // 5. Broadcast splitters
    MPI_Bcast(splitters, P - 1, mpi_tera_type, 0, MPI_COMM_WORLD);

    // 6. Partition local data using splitters
    int *send_counts = calloc(P, sizeof(int));  // Count of elements to send to each process

    // First pass: count how many items go in each bucket
    for (int i = 0; i < local_len; i++) {
        int bucket = 0;
        while (bucket < P - 1 && teraCompare(&local_data[i], &splitters[bucket]) > 0) {
            bucket++;
        }
        send_counts[bucket]++;
    }

    // Compute send displacements
    int *send_displs = malloc(P * sizeof(int));
    send_displs[0] = 0;
    for (int i = 1; i < P; i++) {
        send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
    }

    // Allocate send buffer and temp position tracker
    terarec_t *send_buffer = malloc(local_len * sizeof(terarec_t));
    int *temp_counts = calloc(P, sizeof(int));

    // Second pass: actually copy data to send buffer
    for (int i = 0; i < local_len; i++) {
        int bucket = 0;
        while (bucket < P - 1 && teraCompare(&local_data[i], &splitters[bucket]) > 0) {
            bucket++;
        }
        int pos = send_displs[bucket] + temp_counts[bucket]++;
        send_buffer[pos] = local_data[i];
    }

    free(temp_counts);
    free(splitters);

    // 7. Exchange partitioned values
    int *recv_counts = malloc(P * sizeof(int));
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    int *recv_displs = malloc(P * sizeof(int));
    recv_displs[0] = 0;
    int total_recv = recv_counts[0];
    for (int i = 1; i < P; i++) {
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
        total_recv += recv_counts[i];
    }

    terarec_t *recv_buffer = malloc(total_recv * sizeof(terarec_t));
    MPI_Alltoallv(send_buffer, send_counts, send_displs, mpi_tera_type,
                  recv_buffer, recv_counts, recv_displs, mpi_tera_type,
                  MPI_COMM_WORLD);

    free(send_buffer);
    free(send_counts);
    free(send_displs);
    free(recv_displs);

    // 8. Final local sort
    qsort(recv_buffer, total_recv, sizeof(terarec_t), teraCompare);

    *sorted_data = recv_buffer; // Return values

    int my_count = total_recv;
    MPI_Allgather(&my_count, 1, MPI_INT, sorted_counts, 1, MPI_INT, MPI_COMM_WORLD);

    sorted_displs[0] = 0;
    for (int i = 1; i < P; i++) {
        sorted_displs[i] = sorted_displs[i - 1] + sorted_counts[i - 1];
    }
}
