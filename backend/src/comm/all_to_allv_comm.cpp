/*
 * Function to abstract all to all communication
 * Copyright (c) Sidharth Kumar, et al, see License.md
 */


#include "../parallel_RA_inc.h"
#include <unistd.h>


int TTPL_BT_alltoallv(int n, int r, int bblock, char *sendbuf, int *sendcounts,
									   int *sdispls, MPI_Datatype sendtype, char *recvbuf,
									   int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);


void all_to_all_comm(vector_buffer* vectorized_send_buffer, int vectorized_send_buffer_size, int* send_counts, int *recv_buffer_size, u64 **recv_buffer, MPI_Comm comm)
{
    int nprocs;
    MPI_Comm_size(comm, &nprocs);
    int rank;
    MPI_Comm_rank(comm, &rank);

    int r = ceil(sqrt(nprocs));
    int n = 32;
    int b = 64;

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
//    MPI_Alltoallv(send_buffer, send_counts, send_displacements, MPI_UNSIGNED_LONG_LONG, *recv_buffer, recv_counts, recv_displacements, MPI_UNSIGNED_LONG_LONG, comm);
    TTPL_BT_alltoallv(n, r, b, send_buffer, (char*)send_counts, send_displacements, MPI_UNSIGNED_LONG_LONG, (char*)*recv_buffer, recv_counts, recv_displacements, MPI_UNSIGNED_LONG_LONG, comm);

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
    int rank;
    MPI_Comm_rank(comm, &rank);

    int r = ceil(sqrt(nprocs));
    int n = 32;
    int b = 64;

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

//    MPI_Alltoallv(send_buffer, compute_buffer.cumulative_tuple_process_map, send_disp, MPI_UNSIGNED_LONG_LONG, *recv_buffer, recv_counts, recv_displacements, MPI_UNSIGNED_LONG_LONG, comm);
    TTPL_BT_alltoallv(n, r, b, send_buffer, (char*)send_counts, send_displacements, MPI_UNSIGNED_LONG_LONG, (char*)*recv_buffer, recv_counts, recv_displacements, MPI_UNSIGNED_LONG_LONG, comm);

    delete[] send_buffer;
    delete[] send_disp;
    delete[] recv_displacements;
    delete[] recv_counts;
}


int myPow(int x, unsigned int p) {
  if (p == 0) return 1;
  if (p == 1) return x;

  int tmp = myPow(x, p/2);
  if (p%2 == 0) return tmp * tmp;
  else return x * tmp * tmp;
}


int TTPL_BT_alltoallv(int n, int r, int bblock, char *sendbuf, int *sendcounts,
									   int *sdispls, MPI_Datatype sendtype, char *recvbuf,
									   int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
	if ( r < 2 ) { return -1; }

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	if (r > n) { r = n; }

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	int ngroup, sw;
	int grank, gid, imax, max_sd;
	int local_max_count = 0, max_send_count = 0, id = 0;
	int updated_sentcouts[nprocs], rotate_index_array[nprocs], pos_status[nprocs];
	char *temp_send_buffer, *extra_buffer, *temp_recv_buffer;
	int mpi_errno = MPI_SUCCESS;

	ngroup = nprocs / float(n); // number of groups
    if (r > n) { r = n; }

	sw = ceil(log(n) / float(log(r))); // required digits for intra-Bruck

	grank = rank % n; // rank of each process in a group
	gid = rank / n; // group id
	imax = myPow(r, sw-1) * ngroup;
	max_sd = (ngroup > imax)? ngroup: imax; // max send data block count

	int sent_blocks[max_sd];

	// 1. Find max send elements per data-block
	for (int i = 0; i < nprocs; i++) {
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);

	// 2. create local index array after rotation
	for (int i = 0; i < ngroup; i++) {
		int gsp = i*n;
		for (int j = 0; j < n; j++) {
			rotate_index_array[id++] = gsp + (2 * grank - j + n) % n;
		}
	}

	memset(pos_status, 0, nprocs*sizeof(int));
	memcpy(updated_sentcouts, sendcounts, nprocs*sizeof(int));
	temp_send_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	extra_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	temp_recv_buffer = (char*) malloc(max_send_count*typesize*max_sd);

	// Intra-Bruck
	int spoint = 1, distance = 1, next_distance = r, di = 0;
	for (int x = 0; x < sw; x++) {
		for (int z = 1; z < r; z++) {
			di = 0; spoint = z * distance;
			if (spoint > n - 1) {break;}

			// get the sent data-blocks
			for (int g = 0; g < ngroup; g++) {
				for (int i = spoint; i < n; i += next_distance) {
					for (int j = i; j < (i+distance); j++) {
						if (j > n - 1 ) { break; }
						int id = g*n + (j + grank) % n;
						sent_blocks[di++] = id;
					}
				}
			}

			// 2) prepare metadata and send buffer
			int metadata_send[di];
			int sendCount = 0, offset = 0;
			for (int i = 0; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];
				metadata_send[i] = updated_sentcouts[send_index];

				if (pos_status[send_index] == 0 )
					memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[send_index]*typesize], updated_sentcouts[send_index]*typesize);
				else
					memcpy(&temp_send_buffer[offset], &extra_buffer[sent_blocks[i]*max_send_count*typesize], updated_sentcouts[send_index]*typesize);
				offset += updated_sentcouts[send_index]*typesize;
			}

			int recv_proc = gid*n + (grank + spoint) % n; // receive data from rank + 2^step process
			int send_proc = gid*n + (grank - spoint + n) % n; // send data from rank - 2^k process

			// 3) exchange metadata
			int metadata_recv[di];
			MPI_Sendrecv(metadata_send, di, MPI_INT, send_proc, 0, metadata_recv, di, MPI_INT, recv_proc, 0, comm, MPI_STATUS_IGNORE);

			for(int i = 0; i < di; i++) { sendCount += metadata_recv[i]; }

			// 4) exchange data
			MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, send_proc, 1, temp_recv_buffer, sendCount*typesize, MPI_CHAR, recv_proc, 1, comm, MPI_STATUS_IGNORE);

			// 5) replace
			offset = 0;
			for (int i = 0; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];

				memcpy(&extra_buffer[sent_blocks[i]*max_send_count*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);

				offset += metadata_recv[i]*typesize;
				pos_status[send_index] = 1;
				updated_sentcouts[send_index] = metadata_recv[i];
			}

		}
		distance *= r;
		next_distance *= r;
	}

	// organize data
	int index = 0;
	for (int i = 0; i < nprocs; i++) {
		int d = updated_sentcouts[rotate_index_array[i]]*typesize;
		if (grank == (i % n) ) {
			memcpy(&temp_send_buffer[index], &sendbuf[sdispls[i]*typesize], d);
		}
		else {
			memcpy(&temp_send_buffer[index], &extra_buffer[i*max_send_count*typesize], d);
		}
		index += d;
	}

	free(temp_recv_buffer);
	free(extra_buffer);

	int nsend[ngroup], nrecv[ngroup], nsdisp[ngroup], nrdisp[ngroup];
	int soffset = 0, roffset = 0;
	for (int i = 0; i < ngroup; i++) {
		nsend[i] = 0, nrecv[i] = 0;
		for (int j = 0; j < n; j++) {
			int id = i * n + j;
			int sn = updated_sentcouts[rotate_index_array[id]];
			nsend[i] += sn;
			nrecv[i] += recvcounts[id];
		}
		nsdisp[i] = soffset, nrdisp[i] = roffset;
		soffset += nsend[i] * typesize, roffset += nrecv[i] * typesize;
	}


	if (bblock <= 0 || bblock > ngroup) bblock = ngroup;

	MPI_Request* req = (MPI_Request*)malloc(2*bblock*sizeof(MPI_Request));
	MPI_Status* stat = (MPI_Status*)malloc(2*bblock*sizeof(MPI_Status));
	int req_cnt = 0, ss = 0;

	for (int ii = 0; ii < ngroup; ii += bblock) {
		req_cnt = 0;
		ss = ngroup - ii < bblock ? ngroup - ii : bblock;

		for (int i = 0; i < ss; i++) {
			int nsrc = (gid + i + ii) % ngroup;
			int src =  nsrc * n + grank; // avoid always to reach first master node

			mpi_errno = MPI_Irecv(&recvbuf[nrdisp[nsrc]], nrecv[nsrc]*typesize, MPI_CHAR, src, 0, comm, &req[req_cnt++]);
			if (mpi_errno != MPI_SUCCESS) {return -1;}

		}

		for (int i = 0; i < ss; i++) {
			int ndst = (gid - i - ii + ngroup) % ngroup;
			int dst = ndst * n + grank;

			mpi_errno = MPI_Isend(&temp_send_buffer[nsdisp[ndst]], nsend[ndst]*typesize, MPI_CHAR, dst, 0, comm, &req[req_cnt++]);
			if (mpi_errno != MPI_SUCCESS) {return -1;}
		}

		mpi_errno = MPI_Waitall(req_cnt, req, stat);
		if (mpi_errno != MPI_SUCCESS) {return -1;}
	}

	free(req);
	free(stat);
	free(temp_send_buffer);

	return 0;
}



