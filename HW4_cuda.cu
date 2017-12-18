#include <stdio.h>
#include <stdlib.h>

const int INF = 1000000000;
int V = 20010;
void input(char *inFileName);
void output(char *outFileName);

void block_FW(int B);
int ceil(int a, int b);

int n, m;	// Number of vertices, edges
int* host_ptr = NULL;
size_t pitch;

// for device
int* device_ptr = NULL;
__global__ void gpu_phase1(int* dist, int B, int Round, int block_start_x, int block_start_y, int n, size_t pitch);
__global__ void gpu_phase2(int* dist, int B, int Round, int block_start_x, int block_start_y, int n, int pos, size_t pitch);
__global__ void gpu_phase3(int* dist, int B, int Round, int block_start_x, int block_start_y, int n, size_t pitch);

int main(int argc, char* argv[])
{
	input(argv[1]);
	int B = atoi(argv[3]);

	// allocate memory for device
	cudaMallocPitch(&device_ptr, &pitch, V*sizeof(int), V);
	cudaMemcpy2D(device_ptr, pitch, host_ptr, V*sizeof(int), V*sizeof(int), V, cudaMemcpyHostToDevice);

	block_FW(B);

	cudaMemcpy2D(host_ptr, V*sizeof(int), device_ptr, pitch, V*sizeof(int), V, cudaMemcpyDeviceToHost);
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
	V = n + 10;
	host_ptr = (int*)malloc((size_t)V * V * sizeof(int));

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == j)	host_ptr[i*V+j] = 0;
			else		host_ptr[i*V+j] = INF;
		}
	}

	while (--m >= 0) {
		int a, b, v;
		fscanf(infile, "%d %d %d", &a, &b, &v);
		host_ptr[a*V+b] = v;
	}
    fclose(infile);
}

void output(char *outFileName)
{
	FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
            if (host_ptr[i*V+j] >= INF)
                host_ptr[i*V+j] = INF;
		}
		fwrite(&host_ptr[i*V], sizeof(int), n, outfile);
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
		gpu_phase1<<<blocks, threads, B*B*1*sizeof(int)>>>(device_ptr, B,	r,	r,	r, n, pitch/sizeof(int));

		/* Phase 2*/
		if (r > 0) {
			// left
			blocks = {1, r};
			gpu_phase2<<<blocks, threads, B*B*2*sizeof(int)>>>(device_ptr, B, r,     r,     0, n, 1, pitch/sizeof(int));

			// up
			blocks = {r, 1};
			gpu_phase2<<<blocks, threads, B*B*2*sizeof(int)>>>(device_ptr, B, r,     0,     r, n, 0, pitch/sizeof(int));
		}
		if (r < round - 1) {
			// right
			blocks = {1, round - r -1};
			gpu_phase2<<<blocks, threads, B*B*2*sizeof(int)>>>(device_ptr, B, r,     r,  r +1, n, 1, pitch/sizeof(int));

			// down
			blocks = {round - r -1, 1};
			gpu_phase2<<<blocks, threads, B*B*2*sizeof(int)>>>(device_ptr, B, r,  r +1,     r, n, 0, pitch/sizeof(int));
		}

		/* Phase 3*/
		if (r == 0) {
			// down right
			blocks = {round - r -1, round - r -1};
			gpu_phase3<<<blocks, threads, B*B*3*sizeof(int)>>>(device_ptr, B, r,  r +1,  r +1, n, pitch/sizeof(int));

		}
		else if (r == round - 1) {
			// upper left
			blocks = {r, r};
			gpu_phase3<<<blocks, threads, B*B*3*sizeof(int)>>>(device_ptr, B, r,     0,     0, n, pitch/sizeof(int));
		}
		else {
			// down right
			blocks = {round - r -1, round - r -1};
			gpu_phase3<<<blocks, threads, B*B*3*sizeof(int)>>>(device_ptr, B, r,  r +1,  r +1, n, pitch/sizeof(int));
			// upper left
			blocks = {r, r};
			gpu_phase3<<<blocks, threads, B*B*3*sizeof(int)>>>(device_ptr, B, r,     0,     0, n, pitch/sizeof(int));
			// upper right
			blocks = {r, round -r -1};
			gpu_phase3<<<blocks, threads, B*B*3*sizeof(int)>>>(device_ptr, B, r,     0,  r +1, n, pitch/sizeof(int));
			// down left
			blocks = {round - r -1, r};
			gpu_phase3<<<blocks, threads, B*B*3*sizeof(int)>>>(device_ptr, B, r,  r +1,     0, n, pitch/sizeof(int));
		}
	}
}

extern __shared__ int shared_mem[];
__global__ void gpu_phase1(int* dist, int B, int Round, int block_start_x, int block_start_y, int n, size_t pitch)
{
	int V = pitch;
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
__global__ void gpu_phase2(int* dist, int B, int Round, int block_start_x, int block_start_y, int n, int pos, size_t pitch)
{
	int V = pitch;
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

__global__ void gpu_phase3(int* dist, int B, int Round, int block_start_x, int block_start_y, int n, size_t pitch)
{
	int V = pitch;
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
