/*
 * Function to abstract all to all communication
 * Copyright (c) Sidharth Kumar, et al, see License.md
 */


#include "../parallel_RA_inc.h"
#include <unistd.h>

int twophase_rbruck_alltoallv(int r, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

void all_to_all_comm(vector_buffer* vectorized_send_buffer, int vectorized_send_buffer_size, int* send_counts, int *recv_buffer_size, u64 **recv_buffer, MPI_Comm comm)
{
    int nprocs;
    MPI_Comm_size(comm, &nprocs);
    int rank;
    MPI_Comm_rank(comm, &rank);

    /// send_counts ----> recv_counts
    int* recv_counts = new int[nprocs];
    memset(recv_counts, 0, nprocs * sizeof(int));
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm);

    /// creating send and recv displacements
    int* send_displacements = new int[nprocs];
    int* recv_displacements = new int[nprocs];

    /// creating send_buffer
    u64* send_buffer = new u64[vectorized_send_buffer_size];

    /// Populating send, recv, and data buffer
    recv_displacements[0] = 0;
    send_displacements[0] = 0;
    *recv_buffer_size = recv_counts[0];
    memcpy(send_buffer, (&vectorized_send_buffer[0])->buffer, (&vectorized_send_buffer[0])->size);

    vectorized_send_buffer[0].vector_buffer_free();
    for (int i = 1; i < nprocs; i++)
    {
        send_displacements[i] = send_displacements[i - 1] + send_counts[i - 1];
        recv_displacements[i] = recv_displacements[i - 1] + recv_counts[i - 1];

        *recv_buffer_size = *recv_buffer_size + recv_counts[i];

        memcpy(send_buffer + send_displacements[i], (&vectorized_send_buffer[i])->buffer, (&vectorized_send_buffer[i])->size);
        vectorized_send_buffer[i].vector_buffer_free();
    }
    //std::cout << "vectorized_send_buffer_size" << vectorized_send_buffer_size << std::endl;
    assert(send_displacements[nprocs - 1] + send_counts[nprocs - 1] == vectorized_send_buffer_size);

    /// creating recv_buffer
    *recv_buffer = new u64[*recv_buffer_size];

    /// Actual data transfer
    int r = ceil(sqrt(nprocs));
    twophase_rbruck_alltoallv(r, (char*)send_buffer, send_counts, send_displacements, MPI_UNSIGNED_LONG_LONG, (char*)*recv_buffer, recv_counts, recv_displacements, MPI_UNSIGNED_LONG_LONG, comm);
//    MPI_Alltoallv(send_buffer, send_counts, send_displacements, MPI_UNSIGNED_LONG_LONG, *recv_buffer, recv_counts, recv_displacements, MPI_UNSIGNED_LONG_LONG, comm);
    if (rank == 0)
    	std::cout << "complete twophase_rbruck_alltoallv" << std::endl;
    /// cleanup
    delete[] recv_counts;
    delete[] recv_displacements;
    delete[] send_displacements;
    delete[] send_buffer;

    return;
}



void comm_compaction_all_to_all(all_to_allv_buffer compute_buffer, int **recv_buffer_counts, u64 **recv_buffer, MPI_Comm comm)
{
    u32 RA_count = compute_buffer.ra_count;
    int nprocs = compute_buffer.nprocs;
    int r = ceil(sqrt(nprocs));

    *recv_buffer_counts = new int[RA_count * nprocs];
    memset(*recv_buffer_counts, 0, RA_count * nprocs * sizeof(int));

    MPI_Alltoall(compute_buffer.local_compute_output_count_flat, RA_count, MPI_INT, *recv_buffer_counts, RA_count, MPI_INT, comm);

    int outer_hash_buffer_size = 0;
    int *send_disp = new int[nprocs];
    int *recv_counts = new int[nprocs];
    int *recv_displacements = new int[nprocs];

    recv_counts[0] = 0;
    send_disp[0] = 0;
    recv_displacements[0] = 0;

    u64* send_buffer = new u64[compute_buffer.local_compute_output_size_total];

    u32 boffset = 0;
    int local_max_count = 0;
    int sum0 = 0;
    int local_min_count = compute_buffer.cumulative_tuple_process_map[0];
    for(int i = 0; i < nprocs; i++)
    {
        sum0 = sum0+compute_buffer.cumulative_tuple_process_map[i];
        recv_counts[i] = 0;

        if (compute_buffer.cumulative_tuple_process_map[i] < local_min_count)
            local_min_count = compute_buffer.cumulative_tuple_process_map[i];

        if (compute_buffer.cumulative_tuple_process_map[i] > local_max_count)
            local_max_count = compute_buffer.cumulative_tuple_process_map[i];

        if (i >= 1)
            send_disp[i] = send_disp[i - 1] + compute_buffer.cumulative_tuple_process_map[i - 1];

        for (u32 r = 0; r < RA_count; r++)
        {
            memcpy(send_buffer + boffset, compute_buffer.local_compute_output[r][i].buffer, compute_buffer.local_compute_output[r][i].size);
            boffset = boffset + (compute_buffer.local_compute_output[r][i].size)/sizeof(u64);
            compute_buffer.local_compute_output[r][i].vector_buffer_free();

            recv_counts[i] = recv_counts[i] + (*recv_buffer_counts)[i*RA_count + r] * compute_buffer.width[r];

            assert(compute_buffer.local_compute_output_size_flat[i*RA_count + r] == 
                   compute_buffer.local_compute_output_count_flat[i*RA_count + r] * compute_buffer.width[r]);
        }

        if (i >= 1)
            recv_displacements[i] = recv_displacements[i - 1] + recv_counts[i - 1];
        outer_hash_buffer_size = outer_hash_buffer_size + recv_counts[i];
    }

    *recv_buffer = new u64[outer_hash_buffer_size];

    int max_send_count = 0;
    int average_send_count = 0;
    MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(&local_max_count, &average_send_count, 1, MPI_INT, MPI_SUM, comm);

    twophase_rbruck_alltoallv(r, (char*)send_buffer, compute_buffer.cumulative_tuple_process_map, send_disp, MPI_UNSIGNED_LONG_LONG, (char*)*recv_buffer, recv_counts, recv_displacements, MPI_UNSIGNED_LONG_LONG, comm);
//    MPI_Alltoallv(send_buffer, compute_buffer.cumulative_tuple_process_map, send_disp, MPI_UNSIGNED_LONG_LONG, *recv_buffer, recv_counts, recv_displacements, MPI_UNSIGNED_LONG_LONG, comm);

    delete[] send_buffer;
    delete[] send_disp;
    delete[] recv_displacements;
    delete[] recv_counts;
}


static int myPow(int x, unsigned int p) {
  if (p == 0) return 1;
  if (p == 1) return x;

  int tmp = myPow(x, p/2);
  if (p%2 == 0) return tmp * tmp;
  else return x * tmp * tmp;
}

int twophase_rbruck_alltoallv(int r, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm){

	if ( r < 2 ) { return -1; }

	int rank, nprocs, typesize;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	if ( r > nprocs ) { r = nprocs; }

	int w, nlpow, d;
	int local_max_count=0, max_send_count=0;
	int sendNcopy[nprocs], rotate_index_array[nprocs], pos_status[nprocs];

	MPI_Type_size(sendtype, &typesize);

	w = ceil(log(nprocs) / log(r)); // calculate the number of digits when using r-representation
	nlpow = myPow(r, w-1); // maximum send number of elements
	d = (myPow(r, w) - nprocs) / nlpow; // calculate the number of highest digits

	// 1. Find max send count
	for (int i = 0; i < nprocs; i++) {
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	MPI_Allreduce(&r, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
	memcpy(sendNcopy, sendcounts, nprocs*sizeof(int));
	if (rank == 0)
		std::cout << max_send_count << std::endl;

    // 2. create local index array after rotation
	for (int i = 0; i < nprocs; i++)
		rotate_index_array[i] = (2*rank-i+nprocs)%nprocs;

//	 3. exchange data with log(P) steps
	char* extra_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	char* temp_send_buffer = (char*) malloc(max_send_count*typesize*nlpow);
	char* temp_recv_buffer = (char*) malloc(max_send_count*typesize*nlpow);
	memset(pos_status, 0, nprocs*sizeof(int));

	int sent_blocks[nlpow];
	int di = 0, spoint = 1, distance = myPow(r, w-1), next_distance = distance*r;

	for (int x = w-1; x > -1; x--) {
		int ze = (x == w - 1)? r - d: r;
		for (int z = ze-1; z > 0; z--) {

			// 1) get the sent data-blocks
			di = 0;
			spoint = z * distance;
			for (int i = spoint; i < nprocs; i += next_distance) {
				for (int j = i; j < (i+distance); j++) {
					if (j > nprocs - 1 ) { break; }
					int id = (j + rank) % nprocs;
					sent_blocks[di++] = id;
				}
			}

			// 2) prepare metadata and send buffer
			int metadata_send[di];
			int sendCount = 0, offset = 0;
			for (int i = 0; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];
				metadata_send[i] = sendNcopy[send_index];
				if (pos_status[send_index] == 0)
					memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[send_index]*typesize], sendNcopy[send_index]*typesize);
				else
					memcpy(&temp_send_buffer[offset], &extra_buffer[sent_blocks[i]*max_send_count*typesize], sendNcopy[send_index]*typesize);
				offset += sendNcopy[send_index]*typesize;
			}

			// 3) exchange metadata
			int recvrank = (rank + spoint) % nprocs; // receive data from rank - 2^step process
			int sendrank = (rank - spoint + nprocs) % nprocs; // send data from rank + 2^k process

			int metadata_recv[di];
			MPI_Sendrecv(metadata_send, di, MPI_INT, sendrank, 0, metadata_recv, di, MPI_INT, recvrank, 0, comm, MPI_STATUS_IGNORE);

			for(int i = 0; i < di; i++)
				sendCount += metadata_recv[i];

			// 4) exchange data
			MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, sendrank, 1, temp_recv_buffer, sendCount*typesize, MPI_CHAR, recvrank, 1, comm, MPI_STATUS_IGNORE);

//			// 5) replaces
//			offset = 0;
//			for (int i = 0; i < di; i++) {
//				int send_index = rotate_index_array[sent_blocks[i]];
//
//				int origin_index = (sent_blocks[i] - rank + nprocs) % nprocs;
//				if (origin_index % next_distance == (recvrank - rank + nprocs) % nprocs)
//					memcpy(&recvbuf[rdispls[sent_blocks[i]]*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);
//				else
//					memcpy(&extra_buffer[sent_blocks[i]*max_send_count*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);
//
//				offset += metadata_recv[i]*typesize;
//				pos_status[send_index] = 1;
//				sendNcopy[send_index] = metadata_recv[i];
//			}
		}
		distance /= r;
		next_distance /= r;
	}
//
//	memcpy(&recvbuf[rdispls[rank]*typesize], &sendbuf[sdispls[rank]*typesize], recvcounts[rank]*typesize);

//	free(sendcopy);
	free(temp_send_buffer);
	free(temp_recv_buffer);
	free(extra_buffer);

	return 0;
}


