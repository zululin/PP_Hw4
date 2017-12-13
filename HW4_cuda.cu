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
__global__ void cal_gpu(int* dist, int B, int Round, int block_start_x, int block_start_y, int n);


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

	for (unsigned int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
		/* Phase 1*/
		cal_gpu<<<blocks, threads>>>(D, B,	r,	r,	r, n);

		/* Phase 2*/
		// up
		blocks = {1, r};
		cal_gpu<<<blocks, threads>>>(D, B, r,     r,     0, n);
		// down
		blocks = {1, round - r -1};
		cal_gpu<<<blocks, threads>>>(D, B, r,     r,  r +1, n);
		// left
		blocks = {r, 1};
		cal_gpu<<<blocks, threads>>>(D, B, r,     0,     r, n);
		// right
		blocks = {round - r -1, 1};
		cal_gpu<<<blocks, threads>>>(D, B, r,  r +1,     r, n);

		/* Phase 3*/
		// upper left
		blocks = {r, r};
		cal_gpu<<<blocks, threads>>>(D, B, r,     0,     0, n);
		// down left
		blocks = {r, round -r -1};
		cal_gpu<<<blocks, threads>>>(D, B, r,     0,  r +1, n);
		// upper right
		blocks = {round - r -1, r};
		cal_gpu<<<blocks, threads>>>(D, B, r,  r +1,     0, n);
		// down right
		blocks = {round - r -1, round - r -1};
		cal_gpu<<<blocks, threads>>>(D, B, r,  r +1,  r +1, n);
	}
}

__global__ void cal_gpu(int* dist, int B, int Round, int block_start_x, int block_start_y, int n)
{
	int V = 20010;
	int b_i = block_start_x + blockIdx.x;
	int b_j = block_start_y + blockIdx.y;

	// To calculate B*B elements in the block (b_i, b_j)
	// For each block, it need to compute B times
	for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
		// To calculate original index of elements in the block (b_i, b_j)
		// For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
		int block_internal_start_x 	= b_i * B;
		int block_internal_start_y  = b_j * B;

		int i = block_internal_start_x + threadIdx.x;
		int j = block_internal_start_y + threadIdx.y;

		if (i < n && j < n) {
			if (dist[i*V+k] + dist[k*V+j] < dist[i*V+j]) {
				dist[i*V+j] = dist[i*V+k] + dist[k*V+j];
			}
		}
		__syncthreads();
	}
}
