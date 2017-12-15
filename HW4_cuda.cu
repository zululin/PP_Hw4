#include <stdio.h>
#include <stdlib.h>

const int INF = 1000000000;
const int V = 20010;
void input(char *inFileName);
void output(char *outFileName);

void block_FW(int B);
int ceil(int a, int b);
bool cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m;	// Number of vertices, edges
static int Dist[V][V];

// for device
int* D = NULL;
__global__ void gpu_phase1(int* dist, int B, int Round, int block_start_x, int block_start_y, int n);
__global__ void gpu_phase2(int* dist, int B, int Round, int block_start_x, int block_start_y, int n, int pos);
__global__ void gpu_phase3(int* dist, int B, int Round, int block_start_x, int block_start_y, int n);


int main(int argc, char* argv[])
{
	input(argv[1]);
	int B = atoi(argv[3]);

	// allocate memory for device
	cudaMalloc(&D, (size_t)V * V * sizeof(int));
	cudaMemcpy(D, Dist, (size_t)V * V * sizeof(int), cudaMemcpyHostToDevice);

	block_FW(B);

	cudaMemcpy(Dist, D, (size_t)V * V * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(D);

	output(argv[2]);

	return 0;
}

void input(char *inFileName)
{
	FILE *infile = fopen(inFileName, "r");
	fscanf(infile, "%d %d", &n, &m);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == j)	Dist[i][j] = 0;
			else		Dist[i][j] = INF;
		}
	}

	while (--m >= 0) {
		int a, b, v;
		fscanf(infile, "%d %d %d", &a, &b, &v);
		Dist[a][b] = v;
	}
    fclose(infile);
}

void output(char *outFileName)
{
	FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			// if (Dist[i][j] >= INF)	fprintf(outfile, "INF ");
			// else					fprintf(outfile, "%d ", Dist[i][j]);
            if (Dist[i][j] >= INF)
                Dist[i][j] = INF;
		}
		fwrite(Dist[i], sizeof(int), n, outfile);
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
		gpu_phase1<<<blocks, threads, B*B*1*sizeof(int)>>>(D, B,	r,	r,	r, n);

		/* Phase 2*/
		// up
		blocks = {1, r};
		gpu_phase2<<<blocks, threads, B*B*2*sizeof(int)>>>(D, B, r,     r,     0, n, 1);
		// down
		blocks = {1, round - r -1};
		gpu_phase2<<<blocks, threads, B*B*2*sizeof(int)>>>(D, B, r,     r,  r +1, n, 1);
		// left
		blocks = {r, 1};
		gpu_phase2<<<blocks, threads, B*B*2*sizeof(int)>>>(D, B, r,     0,     r, n, 0);
		// right
		blocks = {round - r -1, 1};
		gpu_phase2<<<blocks, threads, B*B*2*sizeof(int)>>>(D, B, r,  r +1,     r, n, 0);

		/* Phase 3*/
		// upper left
		blocks = {r, r};
		gpu_phase3<<<blocks, threads, B*B*3*sizeof(int)>>>(D, B, r,     0,     0, n);
		// down left
		blocks = {r, round -r -1};
		gpu_phase3<<<blocks, threads, B*B*3*sizeof(int)>>>(D, B, r,     0,  r +1, n);
		// upper right
		blocks = {round - r -1, r};
		gpu_phase3<<<blocks, threads, B*B*3*sizeof(int)>>>(D, B, r,  r +1,     0, n);
		// down right
		blocks = {round - r -1, round - r -1};
		gpu_phase3<<<blocks, threads, B*B*3*sizeof(int)>>>(D, B, r,  r +1,  r +1, n);
	}
}

extern __shared__ int shared_mem[];
__global__ void gpu_phase1(int* dist, int B, int Round, int block_start_x, int block_start_y, int n)
{
	int V = 20010;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tid = ty * B + tx;
	int i = (block_start_x + blockIdx.x) * B + threadIdx.x;
	int j = (block_start_y + blockIdx.y) * B + threadIdx.y;

	// need self block - (b_i, b_j)
	shared_mem[tid] = dist[j*V+i];
	__syncthreads();

	for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
		int k_new = k - B * Round;

		if (i < n && j < n) {
			if (shared_mem[k_new*B+tx] + shared_mem[ty*B+k_new] < shared_mem[tid]) {
				shared_mem[tid] = shared_mem[k_new*B+tx] + shared_mem[ty*B+k_new];
			}
		}
		__syncthreads();
	}

	dist[j*V+i] = shared_mem[tid];
}

extern __shared__ int shared_mem[];
__global__ void gpu_phase2(int* dist, int B, int Round, int block_start_x, int block_start_y, int n, int pos)
{
	int V = 20010;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tid = ty * B + tx;
	int b_i = block_start_x + blockIdx.x;
	int b_j = block_start_y + blockIdx.y;
	int i = b_i * B + threadIdx.x;
	int j = b_j * B + threadIdx.y;

	// need self block - (b_i, b_j) & pivot block - (Round, Round)
	shared_mem[tid+B*B] = dist[j*V+i];
	shared_mem[tid] = dist[(Round*B+ty)*V+(Round*B+tx)];
	__syncthreads();

	for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
		if (i < n && j < n) {
			int k_new_1 = k - B * Round;
			int i_new_1 = i - B * Round;
			int j_new_1 = j - B * Round;

			int k_new_2 = k - B * b_i;
			int j_new_2 = j - B * b_j;

			int i_new_3 = i - B * b_i;
			int k_new_3 = k - B * b_j;

			// up, down
			if (pos == 1) {
				int tmp = shared_mem[k_new_1*B+i_new_1] + shared_mem[j_new_2*B+k_new_2+B*B];

				if (tmp < shared_mem[tid+B*B]) {
					shared_mem[tid+B*B] = tmp;
				}
			}
			// left, right
			else {
				int tmp = shared_mem[k_new_3*B+i_new_3+B*B] + shared_mem[j_new_1*B+k_new_1];

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
	int V = 20010;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tid = ty * B + tx;
	int b_i = block_start_x + blockIdx.x;
	int b_j = block_start_y + blockIdx.y;
	int i = b_i * B + tx;
	int j = b_j * B + ty;

	// need self block - (b_i, b_j) & row / column block
	shared_mem[tid] = dist[j*V+i];
	shared_mem[tid+B*B] = dist[(b_j*B+ty)*V+(Round*B+tx)];			// left&right
	shared_mem[tid+B*B*2] = dist[(Round*B+ty)*V+(b_i*B+tx)];		// up&down
	__syncthreads();


	for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
		if (i < n && j < n) {

			int i_new_1 = i - B * b_i;
			int k_new_1 = k - B * Round;

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
