/*MIT License

Copyright (c) 2019 Xavier Martinez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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
    radiusDic['S'] = 1.80f * factor;
    radiusDic['P'] = 1.80f * factor;
    radiusDic['X'] = 1.40f * factor;
}

// Per-slab streaming callback for the view-dependent path (API_computeSES_view). Called once per
// COMPLETED slab, in priority order (visible/nearest first), with that slab's mesh in HOST memory.
// The pointers are valid only for the duration of the call — the consumer must copy/enqueue and
// return promptly (the surface loop continues right after, so a slow callback stalls the pipeline).
// slabIndex: the slab's index in the priority-ordered sequence. isVisible: 1 if the slab's AABB
// intersects the view frustum, else 0. tris is a flat int array of length NTri (3 per triangle).
typedef void (*SlabMeshCallback)(int slabIndex, int isVisible,
                                 float3 *verts, unsigned int NVert,
                                 int *tris, unsigned int NTri,
                                 int *atomIdPerVert, void *userData);

extern "C" {

    API void API_computeSES(float resoSES, float3 *atomPos, float *atomRad, unsigned int N, float3 *out_vertices,
        unsigned int *NVert, int *out_triangles, unsigned int *NTri, int doSmoothing);

    // View-dependent SES (opt-in; API_computeSES stays the full-surface default).
    // frustumPlanes: 6 planes * 4 floats (nx,ny,nz,d), world space, normals pointing INWARD
    //   (a point p is inside the frustum iff plane.xyz·p + plane.w >= 0 for all 6).
    // camPos: camera world position (for nearest-first ordering).
    // mode: 0 = VISIBLE-ONLY (skip slabs whose AABB is outside the frustum entirely);
    //       1 = VISIBLE-FIRST (compute ALL slabs, but ordered visible+nearest first).
    // slabCb/userData: per-slab streaming callback (may be NULL). When non-NULL, each completed slab
    //   is delivered immediately; the consolidated API_getVertices/Triangles mesh is also built (all
    //   computed slabs) so a caller can still grab one mesh at the end.
    // If frustumPlanes is NULL, behaves like the full computation in offset order (no culling).
    API void API_computeSES_view(float resoSES, float3 *atomPos, float *atomRad, unsigned int N,
        const float *frustumPlanes, float3 camPos, int mode,
        SlabMeshCallback slabCb, void *userData,
        unsigned int *NVert, unsigned int *NTri, int doSmoothing);

    // Two-band distance LOD (opt-in): NEAR band at resoSES (fine), FAR band at resoSES*coarseMul
    // (coarse), selected by complementary frustums (let them overlap slightly for a seam skirt).
    // Keeps the WHOLE surface but computes the distant band cheaper. coarseMul e.g. 2.0. Both bands
    // are streamed via slabCb (near first). Use this when distant-but-visible context should stay
    // shown; use API_computeSES_view mode 0 instead when off-screen parts can be dropped entirely.
    API void API_computeSES_lod(float resoSES, float coarseMul,
        float3 *atomPos, float *atomRad, unsigned int N,
        const float *nearPlanes, const float *farPlanes, float3 camPos,
        SlabMeshCallback slabCb, void *userData,
        unsigned int *NVert, unsigned int *NTri, int doSmoothing);

    API int* API_getTriangles(bool invertTriangles);
    API float3 *API_getVertices();
    API void API_freeMesh();
    API int *API_getAtomIdPerVert();
    // Release the persistent device-buffer pool (call on structure unload / app exit).
    // Device buffers are reused across API_computeSES calls; this frees them.
    API void API_releasePool();
}