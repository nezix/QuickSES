
#define MEASURETIME 1
#define MAX_VERTICES 15


#if defined(__unix__) || defined(__linux__) || defined(__APPLE__) || defined(__MACH__)
#define OS_UNIX
#endif

#if defined(__APPLE__) || defined(__MACH__)
#define OS_OSX
#endif

#if defined(_MSC_VER)
#define OS_WINDOWS
#endif

//
// API export macro
//
#if defined(OS_OSX)
#define API __attribute__((visibility("default")))
#elif defined(OS_WINDOWS)
#define API __declspec(dllexport)
#else
#define API
#endif

std::map<char, float> radiusDic;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void initRadiusDic() {
    float factor = 1.0f;
    radiusDic['O'] = 1.52f * factor;
    radiusDic['C'] = 1.70f * factor;
    radiusDic['N'] = 1.55f * factor;
    radiusDic['H'] = 1.20f * factor;
    radiusDic['S'] = 2.27f * factor;
    radiusDic['P'] = 1.80f * factor;
    radiusDic['X'] = 1.40f * factor;
}

extern "C" {

    API void API_computeSES(float resoSES, float3 *atomPos, float *atomRad, unsigned int N, float3 *out_vertices,
        unsigned int *NVert, int *out_triangles, unsigned int *NTri, int doSmoothing);
    API void API_freeMesh(float3 *verts, int *tris);
    API int* API_getTriangles(bool invertTriangles);
    API float3 *API_getVertices();
    API void API_freeVertices();
    API void API_freeTriangles();
}