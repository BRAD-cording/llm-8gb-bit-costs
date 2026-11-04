// Replicate llama.cpp's allocation order: a ~1.75 GiB model buffer first, then
// the KV block it actually asks for at each context length.
#include <cstdio>
#include <cuda_runtime.h>

static void try_alloc(const char *label, size_t mib) {
    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);
    void *p = nullptr;
    cudaError_t e = cudaMalloc(&p, mib * 1048576ULL);
    if (e != cudaSuccess) {
        printf("%-22s %5zu MiB -> FAIL (%s)  [driver free %.0f MiB]\n",
               label, mib, cudaGetErrorString(e), freeB / 1048576.0);
        cudaGetLastError();
        return;
    }
    cudaMemset(p, 0, mib * 1048576ULL);
    printf("%-22s %5zu MiB -> ok                 [driver free %.0f MiB]\n",
           label, mib, freeB / 1048576.0);
}

int main() {
    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);
    printf("fresh: free %.0f / total %.0f MiB\n", freeB / 1048576.0, totalB / 1048576.0);

    try_alloc("model buffer", 1793);   // Qwen3-1.7B Q8_0 weights
    try_alloc("kv q8_0 @ 4k", 238);
    try_alloc("kv q8_0 @ 16k", 952);
    try_alloc("kv q8_0 @ 32k", 1904);
    try_alloc("kv f16  @ 16k", 1792);
    return 0;
}
