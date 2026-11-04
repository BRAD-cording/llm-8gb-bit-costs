// What can the Orin's CUDA driver actually hand out? cudaMemGetInfo versus
// successive cudaMalloc attempts, printed as it goes.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);
    printf("cudaMemGetInfo: free %.1f MiB / total %.1f MiB\n",
           freeB / 1048576.0, totalB / 1048576.0);

    const size_t steps[] = {64, 128, 256, 512, 768, 1024, 1536, 2048};
    void *blk[8] = {nullptr};
    size_t total = 0;
    for (int i = 0; i < 8; ++i) {
        size_t sz = steps[i] * 1048576ULL;
        cudaError_t e = cudaMalloc(&blk[i], sz);
        if (e != cudaSuccess) {
            printf("  cudaMalloc %4zu MiB -> FAIL (%s)\n", steps[i], cudaGetErrorString(e));
            cudaGetLastError();
            blk[i] = nullptr;
            continue;
        }
        cudaMemset(blk[i], 1, sz);              // touch every page
        total += sz;
        cudaMemGetInfo(&freeB, &totalB);
        printf("  cudaMalloc %4zu MiB -> ok, held %.0f MiB, driver says free %.0f MiB\n",
               steps[i], total / 1048576.0, freeB / 1048576.0);
    }
    for (int i = 0; i < 8; ++i) if (blk[i]) cudaFree(blk[i]);
    return 0;
}
