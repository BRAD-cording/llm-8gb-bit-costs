// Is the 952 MiB refusal caused by page-cache pressure from the mmap'd GGUF?
// memtest2 succeeded at 952 only while the model file was NOT resident.
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>

static void report(const char *when) {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    printf("  [%s] driver free %.0f MiB\n", when, f / 1048576.0);
}

static bool alloc(const char *label, size_t mib) {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    void *p = nullptr;
    cudaError_t e = cudaMalloc(&p, mib * 1048576ULL);
    if (e != cudaSuccess) {
        printf("%-20s %5zu MiB FAIL (%s) [free %.0f]\n", label, mib,
               cudaGetErrorString(e), f / 1048576.0);
        cudaGetLastError();
        return false;
    }
    cudaMemset(p, 0, mib * 1048576ULL);
    printf("%-20s %5zu MiB ok  [free %.0f]\n", label, mib, f / 1048576.0);
    return true;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/brad/c2/models/Qwen3-1.7B-Q8_0.gguf";
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    fstat(fd, &st);
    void *m = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { perror("mmap"); return 1; }

    // Touch as llama.cpp's loader does: read the whole GGUF through the mapping.
    volatile unsigned long sum = 0;
    for (off_t off = 0; off < st.st_size; off += 4096) sum += ((volatile char *)m)[off];
    printf("touched %lld MiB of page cache (sum %lu)\n", (long long)(st.st_size >> 20), sum);
    report("after touch");

    alloc("model buffer", 1793);
    alloc("kv q8_0 @ 4k", 238);
    alloc("kv q8_0 @ 16k", 952);
    alloc("kv q8_0 @ 32k", 1904);
    return 0;
}
