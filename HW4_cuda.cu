#include <stdio.h>
#include <stdlib.h>

const int INF = 1000000000;
void input(char *inFileName);
void output(char *outFileName);

void block_FW(int B);
int ceil(int a, int b);

int n, m;	// Number of vertices, edges
int* host_ptr;

// for device
int* device_ptr = NULL;
__global__ void gpu_phase1(int* dist, int B, int Round, int block_start_x, int block_start_y, int n);
__global__ void gpu_phase2(int* dist, int B, int Round, int block_start_x, int block_start_y, int n, int pos);
__global__ void gpu_phase3(int* dist, int B, int Round, int block_start_x, int block_start_y, int n);

int main(int argc, char* argv[])
{
	input(argv[1]);
	int B = atoi(argv[3]);

	// allocate memory for device
	cudaMalloc(&device_ptr, (size_t)n * n * sizeof(int));
	cudaMemcpy(device_ptr, host_ptr, (size_t)n * n * sizeof(int), cudaMemcpyHostToDevice);

	block_FW(B);

	cudaMemcpy(host_ptr, device_ptr, (size_t)n * n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_ptr);

	output(argv[2]);
	free(host_ptr);

	return 0;
}

void input(char *inFileName)
{
	FILE *infile = fopen(inFileName, "r");
	fscanf(infile, "%d %d", &n, &m);

	// Malloc host memory
	host_ptr = (int*)malloc(n * n * sizeof(int));

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == j)	host_ptr[i*n+j] = 0;
			else		host_ptr[i*n+j] = INF;
		}
	}

	while (--m >= 0) {
		int a, b, v;
		fscanf(infile, "%d %d %d", &a, &b, &v);
		host_ptr[a*n+b] = v;
	}
    fclose(infile);
}

void output(char *outFileName)
{
	FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
            if (host_ptr[i*n+j] >= INF)
                host_ptr[i*n+j] = INF;
		}
		fwrite(&host_ptr[i*n], sizeof(int), n, outfile);
	}
    fclose(outfile);
}

int ceil(int a, int b)
{
	return (a + b -1)/b;
}

void block_FW(int B)
{
	int round = ceil(n, B);
	dim3 blocks = {1, 1};
	dim3 threads = {(unsigned int)B, (unsigned int)B};

	printf("B: %d, Round: %d\n", B, round);

	for (unsigned int r = 0; r < round; ++r) {
		if (r % 10 == 0)
        	printf("%d %d\n", r, round);
		/* Phase 1*/
		blocks = {1, 1};
		gpu_phase1<<<blocks, threads, B*B*1*sizeof(int)>>>(device_ptr, B,	r,	r,	r, n);

		/* Phase 2*/
		// up
		blocks = {1, r};
		gpu_phase2<<<blocks, threads, B*B*2*sizeof(int)>>>(device_ptr, B, r,     r,     0, n, 1);
		// down
		blocks = {1, round - r -1};
		gpu_phase2<<<blocks, threads, B*B*2*sizeof(int)>>>(device_ptr, B, r,     r,  r +1, n, 1);
		// left
		blocks = {r, 1};
		gpu_phase2<<<blocks, threads, B*B*2*sizeof(int)>>>(device_ptr, B, r,     0,     r, n, 0);
		// right
		blocks = {round - r -1, 1};
		gpu_phase2<<<blocks, threads, B*B*2*sizeof(int)>>>(device_ptr, B, r,  r +1,     r, n, 0);

		/* Phase 3*/
		// upper left
		blocks = {r, r};
		gpu_phase3<<<blocks, threads, B*B*3*sizeof(int)>>>(device_ptr, B, r,     0,     0, n);
		// down left
		blocks = {r, round -r -1};
		gpu_phase3<<<blocks, threads, B*B*3*sizeof(int)>>>(device_ptr, B, r,     0,  r +1, n);
		// upper right
		blocks = {round - r -1, r};
		gpu_phase3<<<blocks, threads, B*B*3*sizeof(int)>>>(device_ptr, B, r,  r +1,     0, n);
		// down right
		blocks = {round - r -1, round - r -1};
		gpu_phase3<<<blocks, threads, B*B*3*sizeof(int)>>>(device_ptr, B, r,  r +1,  r +1, n);
	}
}

extern __shared__ int shared_mem[];
__global__ void gpu_phase1(int* dist, int B, int Round, int block_start_x, int block_start_y, int n)
{
	int V = n;
	int tid = threadIdx.y * B + threadIdx.x;
	int i = (block_start_x + blockIdx.x) * B + threadIdx.x;
	int j = (block_start_y + blockIdx.y) * B + threadIdx.y;

	// need self block - (b_i, b_j)
	shared_mem[tid] = dist[j*V+i];
	__syncthreads();

	for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
		if (i < n && j < n) {
			//===== change new posision by: =====//
			// new_i = origin_i - B * B_i
			// new_j = origin_j - B * B_j
			//===================================//
			int k_new = k - B * Round;
			int i_new = i - B * Round;
			int j_new = j - B * Round;

			int tmp = shared_mem[k_new*B+i_new] + shared_mem[j_new*B+k_new];

			if (tmp < shared_mem[tid]) {
				shared_mem[tid] = tmp;
			}
		}
		__syncthreads();
	}
	dist[j*V+i] = shared_mem[tid];
}

extern __shared__ int shared_mem[];
__global__ void gpu_phase2(int* dist, int B, int Round, int block_start_x, int block_start_y, int n, int pos)
{
	int V = n;
	int tid = threadIdx.y * B + threadIdx.x;
	int b_i = block_start_x + blockIdx.x;
	int b_j = block_start_y + blockIdx.y;
	int i = b_i * B + threadIdx.x;
	int j = b_j * B + threadIdx.y;

	// need self block - (b_i, b_j) & pivot block - (Round, Round)
	shared_mem[tid+B*B] = dist[j*V+i];
	shared_mem[tid] = dist[(Round*B+threadIdx.y)*V+(Round*B+threadIdx.x)];
	__syncthreads();

	for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
		if (i < n && j < n) {
			// up, down
			if (pos == 1) {
				int k_new_1 = k - B * Round;
				int i_new_1 = i - B * Round;

				int k_new_2 = k - B * b_i;
				int j_new_2 = j - B * b_j;

				int tmp = shared_mem[k_new_1*B+i_new_1] + shared_mem[j_new_2*B+k_new_2+B*B];

				if (tmp < shared_mem[tid+B*B]) {
					shared_mem[tid+B*B] = tmp;
				}
			}
			// left, right
			else {
				int k_new_1 = k - B * Round;
				int j_new_1 = j - B * Round;

				int i_new_2 = i - B * b_i;
				int k_new_2 = k - B * b_j;

				int tmp = shared_mem[k_new_2*B+i_new_2+B*B] + shared_mem[j_new_1*B+k_new_1];

				if (tmp < shared_mem[tid+B*B]) {
					shared_mem[tid+B*B] = tmp;
				}
			}
		}
		__syncthreads();
	}
	dist[j*V+i] = shared_mem[tid+B*B];
}

__global__ void gpu_phase3(int* dist, int B, int Round, int block_start_x, int block_start_y, int n)
{
	int V = n;
	int tid = threadIdx.y * B + threadIdx.x;
	int b_i = block_start_x + blockIdx.x;
	int b_j = block_start_y + blockIdx.y;
	int i = b_i * B + threadIdx.x;
	int j = b_j * B + threadIdx.y;

	// need self block - (b_i, b_j) & row / column block
	shared_mem[tid] = dist[j*V+i];
	shared_mem[tid+B*B] = dist[(b_j*B+threadIdx.y)*V+(Round*B+threadIdx.x)];			// left, right
	shared_mem[tid+B*B*2] = dist[(Round*B+threadIdx.y)*V+(b_i*B+threadIdx.x)];			// up  , down
	__syncthreads();

	for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
		if (i < n && j < n) {
			// left, right
			int i_new_1 = i - B * b_i;
			int k_new_1 = k - B * Round;

			// up, down
			int k_new_2 = k - B * Round;
			int j_new_2 = j - B * b_j;

			int tmp = shared_mem[k_new_1*B+i_new_1+B*B*2] + shared_mem[j_new_2*B+k_new_2+B*B];

			if (tmp < shared_mem[tid]) {
				shared_mem[tid] = tmp;
			}
		}
		__syncthreads();
	}
	dist[j*V+i] = shared_mem[tid];
}
