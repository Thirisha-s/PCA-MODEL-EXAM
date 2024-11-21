# PCA-MODEL-EXAM

## EXP 01
```[python]
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK(call) { if (cudaError_t err = call) { fprintf(stderr, "Error: %s\n", cudaGetErrorString(err)); exit(1); } }
void initialData(float *data, int size) {
    for (int i = 0; i < size; i++) data[i] = (float)(rand() % 100) / 10.0f;
}
void sumArraysOnHost(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}
__global__ void sumArraysOnGPU(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
int main() {
    int N = 1 << 24; 
    size_t nBytes = N * sizeof(float);
    float *h_A = (float*)malloc(nBytes), *h_B = (float*)malloc(nBytes), *hostRef = (float*)malloc(nBytes), *gpuRef = (float*)malloc(nBytes);
    initialData(h_A, N);
    initialData(h_B, N);
    sumArraysOnHost(h_A, h_B, hostRef, N);
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    int blockSize = 512;
    int numBlocks = (N + blockSize - 1) / blockSize;
    sumArraysOnGPU<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > 1e-8) {
            printf("Mismatch at index %d: host %f, gpu %f\n", i, hostRef[i], gpuRef[i]);
            return -1;
        }
    }
    printf("Arrays match.\n");
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    return 0;
}
```

## EXP 02
```
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call) { if (cudaError_t err = call) { fprintf(stderr, "Error: %s\n", cudaGetErrorString(err)); exit(1); } }

void initialData(int *data, int size) {
    for (int i = 0; i < size; i++) data[i] = rand() & 0xFF;
}

void sumMatrixOnHost(int *A, int *B, int *C, int nx, int ny) {
    for (int i = 0; i < nx * ny; i++) C[i] = A[i] + B[i];
}

__global__ void sumMatrixOnGPU(int *A, int *B, int *C, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny) C[iy * nx + ix] = A[iy * nx + ix] + B[iy * nx + ix];
}

int main() {
    int nx = 1 << 10, ny = 1 << 10;
    int nxy = nx * ny, nBytes = nxy * sizeof(int);

    int *h_A = (int *)malloc(nBytes), *h_B = (int *)malloc(nBytes), *hostRef = (int *)malloc(nBytes), *gpuRef = (int *)malloc(nBytes);
    
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

    int *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, nBytes));
    CHECK(cudaMalloc(&d_B, nBytes));
    CHECK(cudaMalloc(&d_C, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    sumMatrixOnGPU<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < nxy; i++) {
        if (hostRef[i] != gpuRef[i]) {
            printf("Mismatch at %d: host %d, gpu %d\n", i, hostRef[i], gpuRef[i]);
            return -1;
        }
    }

    printf("Arrays match.\n");

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(hostRef); free(gpuRef);

    return 0;
}

```

## EXP 03
```
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace cv;

__global__ void sobelFilter(unsigned char *src, unsigned char *dst, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < w - 1 && y > 0 && y < h - 1) {
        int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int Gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
        int sumX = 0, sumY = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                unsigned char pixel = src[(y + i) * w + (x + j)];
                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }

        int magnitude = min(max(sqrtf(sumX * sumX + sumY * sumY), 0), 255);
        dst[y * w + x] = static_cast<unsigned char>(magnitude);
    }
}

int main() {
    Mat img = imread("image.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) { printf("Error: Image not found.\n"); return -1; }

    int w = img.cols, h = img.rows, size = w * h;
    unsigned char *d_input, *d_output, *h_output = (unsigned char*)malloc(size);
    cudaMalloc(&d_input, size); cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, img.data, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
    sobelFilter<<<grid, block>>>(d_input, d_output, w, h);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    Mat output(h, w, CV_8UC1, h_output);
    imwrite("sobel_output.jpg", output);

    // OpenCV Sobel filter
    Mat opencvOutput;
    auto start = std::chrono::high_resolution_clock::now();
    Sobel(img, opencvOutput, CV_8U, 1, 0);
    auto end = std::chrono::high_resolution_clock::now();
    printf("CUDA time: %f ms\n", std::chrono::duration<double, std::milli>(end - start).count());

    imwrite("sobel_opencv.jpg", opencvOutput);

    free(h_output); cudaFree(d_input); cudaFree(d_output);
    return 0;
}

```

## EXP 04

## WITH MEMSETS:
```
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CHECK(call) { if (call != cudaSuccess) { printf("CUDA Error\n"); exit(1); } }

inline double seconds() {
    struct timeval tp; gettimeofday(&tp, NULL);
    return tp.tv_sec + tp.tv_usec * 1.e-6;
}

void initialData(float *ip, int size) {
    for (int i = 0; i < size; i++) ip[i] = (float)(rand() & 0xFF) / 10.0f;
}

__global__ void sumMatrixGPU(float *A, float *B, float *C, int nx, int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < nx && iy < ny) C[iy * nx + ix] = A[iy * nx + ix] + B[iy * nx + ix];
}

void checkResult(float *hostRef, float *gpuRef, int N) {
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > 1.0E-8) {
            printf("Arrays do not match.\n");
            break;
        }
    }
}

int main(int argc, char **argv) {
    int nx = 1 << (argc > 1 ? atoi(argv[1]) : 12), ny = nx, nxy = nx * ny;
    float *A, *B, *hostRef, *gpuRef;
    CHECK(cudaMallocManaged(&A, nxy * sizeof(float)));
    CHECK(cudaMallocManaged(&B, nxy * sizeof(float)));
    CHECK(cudaMallocManaged(&hostRef, nxy * sizeof(float)));
    CHECK(cudaMallocManaged(&gpuRef, nxy * sizeof(float)));

    initialData(A, nxy); initialData(B, nxy);

    double iStart = seconds();
    sumMatrixGPU<<<(nx + 31)/32, (ny + 31)/32>>>(A, B, gpuRef, nx, ny);
    CHECK(cudaDeviceSynchronize());
    printf("GPU Time: %f sec\n", seconds() - iStart);

    for (int i = 0; i < nxy; i++) hostRef[i] = A[i] + B[i];
    checkResult(hostRef, gpuRef, nxy);

    CHECK(cudaFree(A)); CHECK(cudaFree(B)); CHECK(cudaFree(hostRef)); CHECK(cudaFree(gpuRef));
    return 0;
}
```
## WITHOUT MEMSETS:
```
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define CHECK(call) { if (call != cudaSuccess) { printf("CUDA Error\n"); exit(1); } }

inline double seconds() { struct timeval tp; gettimeofday(&tp, NULL); return tp.tv_sec + tp.tv_usec * 1.e-6; }

void initialData(float *ip, int size) { for (int i = 0; i < size; i++) ip[i] = (float)(rand() & 0xFF) / 10.0f; }

__global__ void sumMatrixGPU(float *A, float *B, float *C, int nx, int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x, iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < nx && iy < ny) C[iy * nx + ix] = A[iy * nx + ix] + B[iy * nx + ix];
}

int main(int argc, char **argv) {
    int nx = 1 << (argc > 1 ? atoi(argv[1]) : 12), ny = nx, nxy = nx * ny;
    float *A, *B, *hostRef, *gpuRef;
    CHECK(cudaMallocManaged(&A, nxy * sizeof(float))); CHECK(cudaMallocManaged(&B, nxy * sizeof(float)));
    CHECK(cudaMallocManaged(&hostRef, nxy * sizeof(float))); CHECK(cudaMallocManaged(&gpuRef, nxy * sizeof(float)));

    initialData(A, nxy); initialData(B, nxy);
    double iStart = seconds();
    sumMatrixGPU<<<(nx + 31)/32, (ny + 31)/32>>>(A, B, gpuRef, nx, ny);
    CHECK(cudaDeviceSynchronize());
    printf("GPU Time: %f sec\n", seconds() - iStart);

    for (int i = 0; i < nxy; i++) hostRef[i] = A[i] + B[i];

    for (int i = 0; i < nxy; i++) if (abs(hostRef[i] - gpuRef[i]) > 1.0E-8) { printf("Mismatch\n"); break; }

    CHECK(cudaFree(A)); CHECK(cudaFree(B)); CHECK(cudaFree(hostRef)); CHECK(cudaFree(gpuRef));
    return 0;
}
```

## EXP 05
```
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

__global__ void bubbleSortKernel(int *d_arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < n - 1; i++) {
        if (idx < n - 1 - i && d_arr[idx] > d_arr[idx + 1]) {
            int temp = d_arr[idx];
            d_arr[idx] = d_arr[idx + 1];
            d_arr[idx + 1] = temp;
        }
        __syncthreads();
    }
}

__device__ void merge(int *arr, int left, int mid, int right, int *temp) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) temp[k++] = arr[i] <= arr[j] ? arr[i++] : arr[j++];
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    for (i = left; i <= right; i++) arr[i] = temp[i];
}

__global__ void mergeSortKernel(int *d_arr, int *d_temp, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int size = 1; size < n; size *= 2) {
        int left = 2 * size * tid;
        if (left < n) {
            int mid = min(left + size - 1, n - 1);
            int right = min(left + 2 * size - 1, n - 1);
            merge(d_arr, left, mid, right, d_temp);
        }
        __syncthreads();
        if (tid == 0) {
            int *temp = d_arr;
            d_arr = d_temp;
            d_temp = temp;
        }
        __syncthreads();
    }
}

void bubbleSort(int *arr, int n, int blockSize, int numBlocks) {
    int *d_arr;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop; 
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    bubbleSortKernel<<<numBlocks, blockSize>>>(d_arr, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    printf("Bubble Sort (GPU): %f ms\n", milliseconds);
}

void mergeSort(int *arr, int n, int blockSize, int numBlocks) {
    int *d_arr, *d_temp;
    cudaMalloc(&d_arr, n * sizeof(int)); cudaMalloc(&d_temp, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop; 
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    mergeSortKernel<<<numBlocks, blockSize>>>(d_arr, d_temp, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr); cudaFree(d_temp);
    printf("Merge Sort (GPU): %f ms\n", milliseconds);
}

void bubbleSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n - 1; i++) 
        for (int j = 0; j < n - i - 1; j++) if (arr[j] > arr[j + 1]) std::swap(arr[j], arr[j + 1]);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    printf("Bubble Sort (CPU): %f ms\n", duration.count());
}

void mergeSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int size = 1; size < n; size *= 2) {
        for (int left = 0; left + size < n; left += 2 * size) {
            int mid = left + size - 1, right = min(left + 2 * size - 1, n - 1);
            merge(arr, left, mid, right, arr + left);  // Merge using host function
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    printf("Merge Sort (CPU): %f ms\n", duration.count());
}

int main() {
    int n_array[] = {500, 1000};
    for (int n : n_array) {
        int *arr = (int*)malloc(n * sizeof(int));
        int blockSize_array[] = {16, 32};
        for (int blockSize : blockSize_array) {
            int numBlocks = (n + blockSize - 1) / blockSize;
            printf("\nArray Size: %d Block Size: %d Num Blocks: %d\n", n, blockSize, numBlocks);

            // Bubble Sort
            for (int i = 0; i < n; i++) arr[i] = rand() % 1000;
            bubbleSortCPU(arr, n);
            bubbleSort(arr, n, blockSize, numBlocks);

            // Merge Sort
            for (int i = 0; i < n; i++) arr[i] = rand() % 1000;
            mergeSortCPU(arr, n);
            mergeSort(arr, n, blockSize, numBlocks);
        }
        free(arr);
    }
    return 0;
}
```

## EXP 06
```
code = """
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define MATRIX_SIZE 1024

void fillMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)(rand() % 100) / 10.0f;
    }
}

void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%0.2f ", matrix[i * cols + j]);
        }
        printf("\\n");
    }
    printf("\\n");
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    fillMatrix(h_A, MATRIX_SIZE, MATRIX_SIZE);
    fillMatrix(h_B, MATRIX_SIZE, MATRIX_SIZE);

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, 
                &alpha, d_A, MATRIX_SIZE, d_B, MATRIX_SIZE, &beta, d_C, MATRIX_SIZE);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Matrix A:\\n");
    printMatrix(h_A, MATRIX_SIZE, MATRIX_SIZE);
    printf("Matrix B:\\n");
    printMatrix(h_B, MATRIX_SIZE, MATRIX_SIZE);
    printf("Matrix C (Result):\\n");
    printMatrix(h_C, MATRIX_SIZE, MATRIX_SIZE);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cublasDestroy(handle);

    return 0;
}
"""
with open('matrix_mul.cu', 'w') as f:
    f.write(code)
```


