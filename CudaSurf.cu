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

#include "cuda_runtime.h"

// includes
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sstream>
#include <iterator>
#include <memory>
#include <map>

// #include <cassert>
#include <fstream>
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>
#include <math.h>

#include <vector>

#include "args.hxx"

#include "Kernels.cu"
#include "cpdb.h"
#include "SmoothMesh.h"
#include "CudaSurf.h"

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include <thrust/sort.h>

using namespace std;

int SLICE = 300;
float probeRadius = PROBERADIUS;
float gridResolutionNeighbor;
float gridResolutionSES = 0.5f;
int laplacianSmoothSteps = 1;
string outputFilePath = "output.obj";
string inputFilePath = "";
bool weldVertices = true;

// ---------------------------------------------------------------------------
// Persistent device-buffer pool.
// QuickSES used to cudaMalloc/cudaFree every device buffer on each API_computeSES
// call. For repeated calls (trajectory playback, re-selection, voxel tweaks) that
// alloc/free churn repeats every frame. A pool keeps each buffer alive across calls
// and only reallocates (grow-only) when a later call needs more than the current
// capacity. Same bytes, same kernels => bit-exact; only the allocator lifetime changes.
// Free everything with API_releasePool() (on structure unload / app exit).
struct DevicePool
{
    void *ptr = NULL;
    size_t capacity = 0; // bytes
};

// One pool per logical buffer (named to match the local it replaces).
static DevicePool poolAtomPosRad;       // sizeof(float4) * N
static DevicePool poolSortedAtomPosRad; // sizeof(float4) * N
static DevicePool poolHashIndex;        // sizeof(int2)   * N
static DevicePool poolCellStartEnd;     // sizeof(int2)   * nbcellsNeighbor
static DevicePool poolGridValues;       // sizeof(float)  * sliceNbCellSES
static DevicePool poolFillCheck;        // sizeof(int)    * sliceNbCellSES
static DevicePool poolVertPerCell;      // sizeof(uint2)  * sliceNbCellSES
static DevicePool poolCompactedVoxels;  // sizeof(uint)   * sliceNbCellSES
static DevicePool poolVertices;         // sizeof(float3) * totalVerts   (output mesh, grow-only)
static DevicePool poolVertOri;          // sizeof(float3) * totalVerts   (weld temp)
static DevicePool poolTri;              // sizeof(int)    * totalVerts   (weld temp)
static DevicePool poolAtomIdPerVert;    // sizeof(int)    * newtotalVerts (weld temp)

// Ensure a pool has at least `bytes` capacity. Grow when too small; also SHRINK when the held
// capacity is far larger than needed, so a huge structure followed by small ones doesn't pin the
// peak allocation until API_releasePool. The 4x hysteresis avoids realloc churn when sizes wobble
// frame-to-frame (a trajectory whose grid stays roughly stable keeps its buffers). Returns the
// device pointer typed as T*.
template <typename T>
static T *ensureDevice(DevicePool &p, size_t bytes)
{
    bool tooSmall = p.capacity < bytes;
    bool tooLarge = bytes > 0 && p.capacity > bytes * 4; // holding >4x what we need now
    if (tooSmall || tooLarge)
    {
        if (p.ptr != NULL)
            gpuErrchk(cudaFree(p.ptr));
        gpuErrchk(cudaMalloc(&p.ptr, bytes));
        p.capacity = bytes;
    }
    return (T *)p.ptr;
}

static void freePool(DevicePool &p)
{
    if (p.ptr != NULL)
    {
        cudaFree(p.ptr);
        p.ptr = NULL;
        p.capacity = 0;
    }
}

// ---------------------------------------------------------------------------
// View-dependent slab culling/ordering support (the API_computeSES_view path).
// A "slab" is one spatial sub-cube of the grid (the offset.x/y/z of the slab loop). For the
// view path we collect the slabs, test each slab's world-space AABB against the camera frustum,
// and process them visible+nearest first (and, in visible-only mode, skip the off-frustum ones).
// Per-slab cost (refine/probe/weld/...) is paid per slab, so skipping a slab saves all of it.
struct ViewParams
{
    bool             enabled = false;        // a frustum was provided
    int              mode    = 1;            // 0 = visible-only, 1 = visible-first
    float            planes[24];             // 6 inward planes * (nx,ny,nz,d)
    float3           camPos = { 0, 0, 0 };
    SlabMeshCallback cb     = NULL;
    void *           userData = NULL;
};

// One pending slab to compute: its grid offset + a sort key (visible flag + distance to camera).
struct SlabTask
{
    int3   offset;        // slab offset (i,j,k) into the SES grid
    float3 aabbCenter;    // world-space center of the slab AABB (for camera distance)
    bool   visible;       // AABB intersects the frustum
    float  camDist2;      // squared distance from camPos to aabbCenter
};

// Test a world-space AABB [bmin,bmax] against 6 inward-pointing frustum planes. Conservative:
// returns false only if the box is fully outside one plane (the standard p-vertex test).
inline bool aabbInFrustum(const float *planes, float3 bmin, float3 bmax)
{
    for (int pl = 0; pl < 6; pl++)
    {
        float nx = planes[pl * 4 + 0], ny = planes[pl * 4 + 1], nz = planes[pl * 4 + 2], d = planes[pl * 4 + 3];
        // p-vertex: the AABB corner farthest along the (inward) normal.
        float px = (nx >= 0.0f) ? bmax.x : bmin.x;
        float py = (ny >= 0.0f) ? bmax.y : bmin.y;
        float pz = (nz >= 0.0f) ? bmax.z : bmin.z;
        if (nx * px + ny * py + nz * pz + d < 0.0f)
            return false; // fully outside this plane => outside the frustum
    }
    return true;
}

unsigned int getMinMax(chain *C, float3 *minVal, float3 *maxVal, float *maxAtom)
{
    atom *A = NULL;
    unsigned int N = 0;

    A = &C->residues[0].atoms[0];
    float3 vmin, vmax, coords;

    vmin.x = vmin.y = vmin.z = 100000.0f;
    vmax.x = vmax.y = vmax.z = -100000.0f;
    *maxAtom = 0.0f;
    while (A != NULL)
    {
        coords = A->coor;
        vmin.x = std::min(vmin.x, coords.x);
        vmin.y = std::min(vmin.y, coords.y);
        vmin.z = std::min(vmin.z, coords.z);

        vmax.x = std::max(vmax.x, coords.x);
        vmax.y = std::max(vmax.y, coords.y);
        vmax.z = std::max(vmax.z, coords.z);

        float atomRad;
        if (radiusDic.count(A->element[0]))
            atomRad = radiusDic[A->element[0]];
        else
            atomRad = radiusDic['X'];
        *maxAtom = std::max(*maxAtom, atomRad);
        N++;
        A = A->next;
    }
    *minVal = vmin;
    *maxVal = vmax;
    return N;
}
unsigned int getMinMax(pdb *P, float3 *minVal, float3 *maxVal, float *maxAtom)
{
    atom *A = NULL;
    unsigned int N = 0;
    chain *C = NULL;
    *maxAtom = 0.0f;
    float3 vmin, vmax, coords;

    vmin.x = vmin.y = vmin.z = 100000.0f;
    vmax.x = vmax.y = vmax.z = -100000.0f;

    for (int chainId = 0; chainId < P->size; chainId++)
    {
        C = &P->chains[chainId];

        A = &C->residues[0].atoms[0];

        while (A != NULL)
        {
            coords = A->coor;
            vmin.x = std::min(vmin.x, coords.x);
            vmin.y = std::min(vmin.y, coords.y);
            vmin.z = std::min(vmin.z, coords.z);

            vmax.x = std::max(vmax.x, coords.x);
            vmax.y = std::max(vmax.y, coords.y);
            vmax.z = std::max(vmax.z, coords.z);

            float atomRad;
            if (radiusDic.count(A->element[0]))
                atomRad = radiusDic[A->element[0]];
            else
                atomRad = radiusDic['X'];
            *maxAtom = std::max(*maxAtom, atomRad);
            N++;
            A = A->next;
        }
    }
    *minVal = vmin;
    *maxVal = vmax;
    return N;
}
void getMinMax(float3 *positions, float *radii, unsigned int N, float3 *minVal, float3 *maxVal, float *maxAtom)
{
    *maxAtom = 0.0f;
    float3 vmin, vmax, coords;

    vmin.x = vmin.y = vmin.z = 100000.0f;
    vmax.x = vmax.y = vmax.z = -100000.0f;

    for (unsigned int a = 0; a < N; a++)
    {
        coords = positions[a];
        vmin.x = std::min(vmin.x, coords.x);
        vmin.y = std::min(vmin.y, coords.y);
        vmin.z = std::min(vmin.z, coords.z);

        vmax.x = std::max(vmax.x, coords.x);
        vmax.y = std::max(vmax.y, coords.y);
        vmax.z = std::max(vmax.z, coords.z);

        float atomRad = radii[a];

        *maxAtom = std::max(*maxAtom, atomRad);
    }
    *minVal = vmin;
    *maxVal = vmax;
}

float4 *getArrayAtomPosRad(chain *C, unsigned int N)
{

    float4 *result = new float4[N];
    atom *A = NULL;
    int id = 0;

    A = &C->residues[0].atoms[0];
    float3 coords;
    while (A != NULL)
    {
        coords = A->coor;

        float atomRad = radiusDic[A->element[0]];
        result[id].x = coords.x;
        result[id].y = coords.y;
        result[id].z = coords.z;
        result[id].w = atomRad;
        id++;
        A = A->next;
    }

    return result;
}

float4 *getArrayAtomPosRad(pdb *P, unsigned int N)
{
    chain *C = NULL;
    atom *A = NULL;
    float4 *result = new float4[N];
    int id = 0;

    for (int chainId = 0; chainId < P->size; chainId++)
    {
        C = &P->chains[chainId];

        A = &C->residues[0].atoms[0];
        float3 coords;
        while (A != NULL)
        {
            coords = A->coor;

            float atomRad = radiusDic[A->element[0]];
            result[id].x = coords.x;
            result[id].y = coords.y;
            result[id].z = coords.z;
            result[id].w = atomRad;
            id++;
            A = A->next;
        }
    }

    return result;
}

float4 *getArrayAtomPosRad(float3 *positions, float *radii, unsigned int N)
{
    float4 *result = (float4 *)malloc(sizeof(float4) * N);
    int id = 0;

    for (int a = 0; a < N; a++)
    {
        float3 coords = positions[a];
        float atomRad = radii[a];
        result[id].x = coords.x;
        result[id].y = coords.y;
        result[id].z = coords.z;
        result[id].w = atomRad;
        id++;
    }

    return result;
}

float computeMaxDist(float3 minVal, float3 maxVal, float maxAtomRad)
{
    return std::max(maxVal.x - minVal.x, std::max(maxVal.y - minVal.y, maxVal.z - minVal.z)) + (2 * maxAtomRad) + (4 * probeRadius);
}

void writeToObj(const string &fileName, const vector<int> &meshTriSizes, const vector<int> &meshVertSizes,
                const vector<float3 *> &Allvertices, const vector<int3 *> &AllTriangles)
{

#if MEASURETIME
    std::clock_t start = std::clock();
#endif

    FILE *fptr;
    if ((fptr = fopen(fileName.c_str(), "w")) == NULL)
    {
        fprintf(stderr, "Failed to open output file\n");
        exit(-1);
    }
    for (int m = 0; m < meshTriSizes.size(); m++)
    {

        for (int i = 0; i < meshVertSizes[m]; i++)
        {
            float3 vert = Allvertices[m][i];
            fprintf(fptr, "v %.3f %.3f %.3f\n", vert.x, vert.y, vert.z);
        }
    }

    fprintf(fptr, "\n");
    unsigned int cumulMesh = 0;
    for (int m = 0; m < meshTriSizes.size(); m++)
    {
        int ntri = meshTriSizes[m];
        for (int i = 0; i < ntri; i++)
        {
            int3 triangle = AllTriangles[m][i];
            fprintf(fptr, "f %d %d %d\n", cumulMesh + triangle.y + 1, cumulMesh + triangle.x + 1, cumulMesh + triangle.z + 1);
        }
        cumulMesh += meshVertSizes[m];
    }

    fclose(fptr);

#if MEASURETIME
    std::cerr << "Time for writting " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
#endif
}

void writeToObj(const string &fileName, const MeshData &mesh)
{
#if MEASURETIME
    std::clock_t start = std::clock();
#endif

    FILE *fptr;
    if ((fptr = fopen(fileName.c_str(), "w")) == NULL)
    {
        fprintf(stderr, "Failed to open output file\n");
        exit(-1);
    }

    for (int i = 0; i < mesh.NVertices; i++)
    {
        float3 vert = mesh.vertices[i];
        fprintf(fptr, "v %.3f %.3f %.3f\n", vert.x, vert.y, vert.z);
    }

    fprintf(fptr, "\n");
    for (int i = 0; i < mesh.NTriangles; i++)
    {
        int3 triangle = mesh.triangles[i];
        fprintf(fptr, "f %d %d %d\n", triangle.y + 1, triangle.x + 1, triangle.z + 1);
    }
    fclose(fptr);
#if MEASURETIME
    std::cerr << "Time for writting " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
#endif
}
void writeToObj(const string &fileName, std::vector<MeshData> meshes)
{

#if MEASURETIME
    std::clock_t start = std::clock();
#endif

    FILE *fptr;
    if ((fptr = fopen(fileName.c_str(), "w")) == NULL)
    {
        fprintf(stderr, "Failed to open output file\n");
        exit(-1);
    }
    unsigned int cumulVert = 0;
    for (int m = 0; m < meshes.size(); m++)
    {
        MeshData mesh = meshes[m];

        // smoothMeshLaplacian(2, mesh);

        for (int i = 0; i < mesh.NVertices; i++)
        {
            float3 vert = mesh.vertices[i];
            fprintf(fptr, "v %.3f %.3f %.3f\n", vert.x, vert.y, vert.z);
        }
    }
    fprintf(fptr, "\n");
    for (int m = 0; m < meshes.size(); m++)
    {
        MeshData mesh = meshes[m];

        for (int i = 0; i < mesh.NTriangles; i++)
        {
            int3 triangle = mesh.triangles[i];
            fprintf(fptr, "f %d %d %d\n", cumulVert + triangle.y + 1, cumulVert + triangle.x + 1, cumulVert + triangle.z + 1);
        }
        cumulVert += mesh.NVertices;
    }
    fclose(fptr);
#if MEASURETIME
    std::cerr << "Time for writting " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
#endif
}

MeshData computeMarchingCubes(int3 sliceGridSESDim, int cutMC, int sliceNbCellSES, float *cudaGridValues, uint2 *vertPerCell,
                              unsigned int *compactedVoxels, int3 gridSESDim, float4 originGridSESDx, int3 offset, float4 *cudaSortedAtomPosRad,
                              int2 *cellStartEnd, int3 gridNeighborDim, float4 originGridNeighborDx, int rangeSearchRefine)
{

    unsigned long int memAlloc = 0;
    memsetCudaUInt2<<<(sliceNbCellSES + NBTHREADS - 1) / NBTHREADS, NBTHREADS>>>(vertPerCell, make_uint2(0, 0), sliceNbCellSES);

    MeshData result;
    float iso = 0.0f;
    dim3 localWorkSize(cutMC, cutMC, cutMC);
    dim3 globalWorkSize((sliceGridSESDim.x + cutMC - 1) / cutMC, (sliceGridSESDim.y + cutMC - 1) / cutMC, (sliceGridSESDim.z + cutMC - 1) / cutMC);

    countVertexPerCell<<<globalWorkSize, localWorkSize>>>(iso, sliceGridSESDim, cudaGridValues, vertPerCell, rangeSearchRefine, offset);
    gpuErrchk(cudaPeekAtLastError());

    uint2 lastElement, lastScanElement;
    gpuErrchk(cudaMemcpy((void *)&lastElement, (void *)(vertPerCell + sliceNbCellSES - 1), sizeof(uint2), cudaMemcpyDeviceToHost));

    thrust::exclusive_scan(thrust::device_ptr<uint2>(vertPerCell),
                           thrust::device_ptr<uint2>(vertPerCell + sliceNbCellSES),
                           thrust::device_ptr<uint2>(vertPerCell),
                           make_uint2(0, 0), add_uint2());

    gpuErrchk(cudaMemcpy((void *)&lastScanElement, (void *)(vertPerCell + sliceNbCellSES - 1), sizeof(uint2), cudaMemcpyDeviceToHost));

    unsigned int totalVoxels = lastElement.y + lastScanElement.y;
    unsigned int totalVerts = lastElement.x + lastScanElement.x;

    // Pooled output-vertex buffer (grow-only; reused across slabs and across calls).
    // Slabs run sequentially within a call, so a single pooled buffer is safe.
    float3 *cudaVertices = ensureDevice<float3>(poolVertices, sizeof(float3) * totalVerts);
    memAlloc += sizeof(float3) * totalVerts;
    // generateTriangleVerticesSMEM guards writes with `index < totalVerts-3`, leaving the last
    // up-to-3 slots UNWRITTEN; the weld (groupVertices+sort+unique) then reads all totalVerts, so
    // those tail slots must be deterministic. Fresh cudaMalloc happened to hand back zeroed pages;
    // the pool reuses a buffer holding the previous slab's data, which perturbed the welded count
    // (1AON v0.5: 1951253 -> 1951255). Zero the buffer so the tail is always (0,0,0) regardless of
    // allocator -> deterministic AND pool-safe (also removes the pre-existing latent nondeterminism).
    gpuErrchk(cudaMemset(cudaVertices, 0, sizeof(float3) * totalVerts));

    globalWorkSize = dim3((sliceGridSESDim.x + localWorkSize.x - 1) / localWorkSize.x, (sliceGridSESDim.y + localWorkSize.y - 1) / localWorkSize.y, (sliceGridSESDim.z + localWorkSize.z - 1) / localWorkSize.z);

    compactVoxels<<<globalWorkSize, localWorkSize>>>(compactedVoxels, vertPerCell, lastElement.y, sliceNbCellSES, sliceNbCellSES + 1, sliceGridSESDim, rangeSearchRefine, offset);
    gpuErrchk(cudaPeekAtLastError());

    unsigned int totalVoxsqr3 = (unsigned int)ceil((totalVoxels + NBTHREADS - 1) / NBTHREADS);

    if (totalVoxsqr3 == 0)
    {
        return result;
    }

    globalWorkSize = dim3(totalVoxsqr3, 1, 1);

    generateTriangleVerticesSMEM<<<globalWorkSize, NBTHREADS>>>(cudaVertices, compactedVoxels, vertPerCell, cudaGridValues, originGridSESDx,
                                                                iso, totalVoxels, totalVerts - 3, sliceGridSESDim, offset);

    gpuErrchk(cudaPeekAtLastError());

    if (weldVertices)
    {
        // Weld vertices
        float3 *vertOri;
        int *cudaTri;
        int *cudaAtomIdPerVert;
        unsigned int newtotalVerts = totalVerts;

        int global = (unsigned int)ceil((totalVerts + NBTHREADS - 1) / NBTHREADS);
        groupVertices<<<global, NBTHREADS>>>(cudaVertices, totalVerts, EPSILON);
        gpuErrchk(cudaPeekAtLastError());

        vertOri = ensureDevice<float3>(poolVertOri, sizeof(float3) * totalVerts);
        gpuErrchk(cudaMemcpy(vertOri, cudaVertices, sizeof(float3) * totalVerts, cudaMemcpyDeviceToDevice));
        cudaTri = ensureDevice<int>(poolTri, sizeof(int) * totalVerts);

        memAlloc += sizeof(float3) * totalVerts;
        memAlloc += sizeof(int) * totalVerts;

        thrust::device_ptr<float3> d_vertThrust = thrust::device_pointer_cast(cudaVertices);
        thrust::device_ptr<vec3> vertThrust((vec3 *)thrust::raw_pointer_cast(d_vertThrust));

        thrust::sort(vertThrust, vertThrust + totalVerts);

        thrust::device_ptr<vec3> last = thrust::unique(vertThrust, vertThrust + totalVerts);

        newtotalVerts = last - vertThrust;

        thrust::device_ptr<float3> d_vertOriThrust(vertOri);
        thrust::device_ptr<vec3> vertOriThrust((vec3 *)thrust::raw_pointer_cast(d_vertOriThrust));

        thrust::device_ptr<int> triThrust(cudaTri);
        thrust::lower_bound(vertThrust, last, vertOriThrust, vertOriThrust + totalVerts, triThrust);
        gpuErrchk(cudaPeekAtLastError());

        cudaAtomIdPerVert = ensureDevice<int>(poolAtomIdPerVert, sizeof(int) * newtotalVerts);
        memAlloc += sizeof(int) * newtotalVerts;

        global = (unsigned int)ceil((newtotalVerts + NBTHREADS - 1) / NBTHREADS);

        // Look for atoms around vertices => could be done a way smarter way during the MC step
        closestAtomPerVertex<<<global, NBTHREADS>>>(cudaAtomIdPerVert, cudaVertices, newtotalVerts, gridNeighborDim,
                                                    originGridNeighborDx, originGridSESDx, cellStartEnd, cudaSortedAtomPosRad);

        gpuErrchk(cudaPeekAtLastError());

        cerr << "MC allocation = " << memAlloc / 1000000.0f << " Mo" << endl;

        int Ntriangles = totalVerts / 3;

        result.vertices = (float3 *)malloc(sizeof(float3) * newtotalVerts);
        result.triangles = (int3 *)malloc(sizeof(int3) * Ntriangles);
        result.atomIdPerVert = (int *)malloc(sizeof(int) * newtotalVerts);
        result.NVertices = newtotalVerts;
        result.NTriangles = Ntriangles;

        int *tmpTri = (int *)malloc(sizeof(int) * totalVerts);

        gpuErrchk(cudaMemcpy(result.vertices, cudaVertices, sizeof(float3) * newtotalVerts, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(result.atomIdPerVert, cudaAtomIdPerVert, sizeof(int) * newtotalVerts, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(tmpTri, cudaTri, sizeof(int) * totalVerts, cudaMemcpyDeviceToHost));

        // Store the triangle in a 3d vector
        for (int i = 0; i < Ntriangles; i++)
        {
            result.triangles[i].x = tmpTri[i * 3 + 0];
            result.triangles[i].y = tmpTri[i * 3 + 1];
            result.triangles[i].z = tmpTri[i * 3 + 2];
        }
        free(tmpTri);

        // cudaVertices / vertOri / cudaTri / cudaAtomIdPerVert are pooled now
        // (reused across slabs + calls, freed by API_releasePool). The old non-welded
        // path leaked cudaVertices; pooling fixes that leak too.
    }
    else
    {
        int Ntriangles = totalVerts / 3;

        result.vertices = (float3 *)malloc(sizeof(float3) * totalVerts);
        result.triangles = (int3 *)malloc(sizeof(int3) * Ntriangles);
        result.atomIdPerVert = (int *)malloc(sizeof(int) * totalVerts);
        result.NVertices = totalVerts;
        result.NTriangles = Ntriangles;

        gpuErrchk(cudaMemcpy(result.vertices, cudaVertices, sizeof(float3) * totalVerts, cudaMemcpyDeviceToHost));

        for (int i = 0; i < Ntriangles; i++)
        {
            result.triangles[i].x = i * 3 + 0;
            result.triangles[i].y = i * 3 + 1;
            result.triangles[i].z = i * 3 + 2;
        }
    }

    return result;
}

std::vector<MeshData> computeSlicedSES(float3 positions[], float radii[], unsigned int N, float resoSES, int doSmoothing = 1,
                                       const ViewParams *view = NULL)
{
#if MEASURETIME
    std::clock_t startSES = std::clock();
#endif

    // TODO (session API): a future explicit API_beginSession/computeFrame/endSession could let the
    // caller assert "same molecule, only positions moved" to cache grid dims/bounds and skip
    // re-derivation. Measured not worth a transparent version: getMinMax + dim setup are O(N) host
    // work, microseconds for small N and negligible vs GPU compute for large N; the per-frame cost is
    // the GPU hash/sort/grid/MC, which must rerun since the surface moves. The buffer pool (above) is
    // the part of the trajectory win that pays off.

    // Record a mesh per slice
    std::vector<MeshData> resultMeshes;

    float3 minVal, maxVal;
    float maxAtomRad = 0.0;

    getMinMax(positions, radii, N, &minVal, &maxVal, &maxAtomRad);

    cerr << "#atoms : " << N << endl;
    if (N <= 1)
    {
        cerr << "Failed to parse the PDB or empty PDB file" << endl;
        return resultMeshes;
    }

    float4 *atomPosRad = getArrayAtomPosRad(positions, radii, N);
    float maxDist = computeMaxDist(minVal, maxVal, maxAtomRad);

    gridResolutionNeighbor = probeRadius + maxAtomRad;

    // Grid is a cube
    float3 originGridNeighbor = {
        minVal.x - maxAtomRad - probeRadius,
        minVal.y - maxAtomRad - probeRadius,
        minVal.z - maxAtomRad - probeRadius};

    int gridNeighborSize = (int)ceil(maxDist / gridResolutionNeighbor);

    int3 gridNeighborDim = {gridNeighborSize, gridNeighborSize, gridNeighborSize};

    int gridSESSize = (int)ceil(maxDist / resoSES);

    int3 gridSESDim = {gridSESSize, gridSESSize, gridSESSize};

    float4 originGridNeighborDx = {
        originGridNeighbor.x,
        originGridNeighbor.y,
        originGridNeighbor.z,
        gridResolutionNeighbor};

    float4 originGridSESDx = {
        originGridNeighborDx.x,
        originGridNeighborDx.y,
        originGridNeighborDx.z,
        resoSES};

    unsigned int nbcellsNeighbor = gridNeighborDim.x * gridNeighborDim.y * gridNeighborDim.z;
    // unsigned int nbcellsSES = gridSESDim.x * gridSESDim.y * gridSESDim.z;

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    float4 *cudaAtomPosRad;
    float4 *cudaSortedAtomPosRad;
    int2 *cudaHashIndex;
    int2 *cellStartEnd;
    float *cudaGridValues;
    int *cudaFillCheck;

    // Marching cubes data
    uint2 *vertPerCell;
    unsigned int *compactedVoxels;

    // Pooled (reused across calls; grow-only). See DevicePool above.
    cudaAtomPosRad = ensureDevice<float4>(poolAtomPosRad, sizeof(float4) * N);
    cudaSortedAtomPosRad = ensureDevice<float4>(poolSortedAtomPosRad, sizeof(float4) * N);
    cudaHashIndex = ensureDevice<int2>(poolHashIndex, sizeof(int2) * N);
    cellStartEnd = ensureDevice<int2>(poolCellStartEnd, sizeof(int2) * nbcellsNeighbor);

    //-------------- Step 1 : Insert atoms in neighbor cells -----------------

    // Copy atom positions and radii to GPU
    gpuErrchk(cudaMemcpy(cudaAtomPosRad, atomPosRad, sizeof(float4) * N, cudaMemcpyHostToDevice));

    // Compute atom cell ids
    hashAtoms<<<(N + NBTHREADS - 1) / NBTHREADS, NBTHREADS>>>(N, cudaAtomPosRad, gridNeighborDim, originGridNeighborDx, cudaHashIndex, N);

    gpuErrchk(cudaPeekAtLastError());

    // Sort atoms cell id
    compare_int2 cmp;
    thrust::device_ptr<int2> D_beg = thrust::device_pointer_cast(cudaHashIndex);
    thrust::sort(D_beg, D_beg + N, cmp);
    gpuErrchk(cudaPeekAtLastError());

    memsetCudaInt2<<<(nbcellsNeighbor + NBTHREADS - 1) / NBTHREADS, NBTHREADS>>>(cellStartEnd, make_int2(EMPTYCELL, EMPTYCELL), nbcellsNeighbor);

    // Reorder atoms positions and radii and fill cellStartEnd
    sortCell<<<(N + NBTHREADS - 1) / NBTHREADS, NBTHREADS>>>(N, cudaAtomPosRad, cudaHashIndex, cudaSortedAtomPosRad, cellStartEnd);

    gpuErrchk(cudaPeekAtLastError());

    // cudaAtomPosRad is pooled now (freed by API_releasePool, not per-call).

    // std::cerr << "Time for setup " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    // start = std::clock();

    //-------------- Step 2 : Compute points of the grid outside or inside the surface -----------------
    // Use slices of the grid to avoid allocating large amount of data
    int rangeSearchRefine = (int)ceil(PROBERADIUS / resoSES);
    int sliceSmallSize = min(SLICE, gridSESSize);
    int sliceSize = min(SLICE + 2 * rangeSearchRefine, gridSESSize);
    // int sliceSmallNbCellSES = sliceSmallSize * sliceSmallSize * sliceSmallSize;
    int sliceNbCellSES = sliceSize * sliceSize * sliceSize;
    // int3 sliceGridSESDim = make_int3(sliceSmallSize, sliceSmallSize, sliceSmallSize);
    int3 fullSliceGridSESDim = make_int3(sliceSize, sliceSize, sliceSize);

    // Pooled (reused across calls; grow-only). See DevicePool above.
    cudaGridValues = ensureDevice<float>(poolGridValues, sizeof(float) * sliceNbCellSES);
    cudaFillCheck = ensureDevice<int>(poolFillCheck, sizeof(int) * sliceNbCellSES);

    vertPerCell = ensureDevice<uint2>(poolVertPerCell, sizeof(uint2) * sliceNbCellSES);
    compactedVoxels = ensureDevice<unsigned int>(poolCompactedVoxels, sizeof(unsigned int) * sliceNbCellSES);

    gpuErrchk(cudaPeekAtLastError());

    cerr << "Allocating " << (((sizeof(int) + sizeof(float)) * sliceNbCellSES + 3 * sizeof(int) * sliceNbCellSES) + 2 * sizeof(float4) * N + sizeof(int2) * N + sizeof(int2) * nbcellsNeighbor) / 1000000.0f << " Mo" << endl;

    int3 offset = {0, 0, 0};
    int cut = 8;

    cerr << "Full size grid = " << gridSESSize << " x " << gridSESSize << " x " << gridSESSize << endl;

    // Streams for the refine kernel are reused across all slabs: create once here and destroy once
    // after the slab loop, instead of per-slab create/destroy (which ran 8x on large multi-slab
    // grids). Stream handles are slab-independent, so hoisting is correctness-neutral.
    const int nbStream = 4;
    cudaStream_t streams[nbStream];
    for (int si = 0; si < nbStream; si++)
        cudaStreamCreate(&(streams[si]));

    // ---- Build the slab work-list, then (for the view path) cull + order it ----
    // Each slab is a spatial sub-cube at offset (i,j,k). For the view path we compute its world-space
    // AABB, frustum-test it, and process visible+nearest first; visible-only mode drops off-frustum
    // slabs (skipping ALL their per-slab cost). Without a view, this is the original offset order.
    float3 originGridNeighbor_v = make_float3(originGridNeighborDx.x, originGridNeighborDx.y, originGridNeighborDx.z);
    std::vector<SlabTask> slabTasks;
    for (int i = 0; i < gridSESSize; i += sliceSmallSize)
        for (int j = 0; j < gridSESSize; j += sliceSmallSize)
            for (int k = 0; k < gridSESSize; k += sliceSmallSize)
            {
                SlabTask t;
                t.offset = make_int3(i, j, k);
                // Slab world-space AABB: the processed slab spans [offset, offset+sliceSmallSize) cells
                // of edge dxSES, from the grid origin. (sliceSmallSize is the non-halo stride.)
                float3 bmin = gridToSpace(make_int3(i, j, k), originGridNeighbor_v, resoSES);
                float3 bmax = gridToSpace(make_int3(min(i + sliceSmallSize, gridSESSize),
                                                    min(j + sliceSmallSize, gridSESSize),
                                                    min(k + sliceSmallSize, gridSESSize)),
                                          originGridNeighbor_v, resoSES);
                t.aabbCenter = make_float3((bmin.x + bmax.x) * 0.5f, (bmin.y + bmax.y) * 0.5f, (bmin.z + bmax.z) * 0.5f);
                if (view != NULL && view->enabled)
                {
                    t.visible = aabbInFrustum(view->planes, bmin, bmax);
                    float ddx = t.aabbCenter.x - view->camPos.x;
                    float ddy = t.aabbCenter.y - view->camPos.y;
                    float ddz = t.aabbCenter.z - view->camPos.z;
                    t.camDist2 = ddx * ddx + ddy * ddy + ddz * ddz;
                }
                else { t.visible = true; t.camDist2 = 0.0f; }
                // visible-only mode (0): drop off-frustum slabs entirely.
                if (view != NULL && view->enabled && view->mode == 0 && !t.visible)
                    continue;
                slabTasks.push_back(t);
            }
    // Order: visible first, then nearest-camera first. Stable w.r.t. the original order otherwise.
    if (view != NULL && view->enabled)
        std::stable_sort(slabTasks.begin(), slabTasks.end(), [](const SlabTask &a, const SlabTask &b) {
            if (a.visible != b.visible) return a.visible > b.visible; // visible (true) first
            return a.camDist2 < b.camDist2;                           // nearest first
        });

    // cudaEventRecord(start);
    {
        for (size_t taskId = 0; taskId < slabTasks.size(); taskId++)
        {
            const SlabTask &task = slabTasks[taskId];
            {
                offset.x = task.offset.x;
                offset.y = task.offset.y;
                offset.z = task.offset.z;
                // cerr << "-----------------------------\nStarting : " << offset.x << " / " << offset.y << " / " << offset.z << endl;

                memsetCudaFloat<<<(sliceNbCellSES + NBTHREADS - 1) / NBTHREADS, NBTHREADS>>>(cudaGridValues, probeRadius, sliceNbCellSES);
                memsetCudaInt<<<(sliceNbCellSES + NBTHREADS - 1) / NBTHREADS, NBTHREADS>>>(cudaFillCheck, EMPTYCELL, sliceNbCellSES);

                dim3 localWorkSize(cut, cut, cut);
                // dim3 globalWorkSize((sliceSmallSize + cut - 1) / cut, (sliceSmallSize + cut - 1) / cut, (sliceSmallSize + cut - 1) / cut);
                dim3 globalWorkSize((sliceSize + cut - 1) / cut, (sliceSize + cut - 1) / cut, (sliceSize + cut - 1) / cut);

                int3 reducedOffset = make_int3(max(0, offset.x - rangeSearchRefine),
                                               max(0, offset.y - rangeSearchRefine),
                                               max(0, offset.z - rangeSearchRefine));

                // int3 reducedOffset = offset;

                // cerr << "Fulllll : " << fullSliceGridSESDim.x << ", " << fullSliceGridSESDim.y << ", " << fullSliceGridSESDim.z << endl;
                // cerr << "global = " << globalWorkSize.x << ", " << globalWorkSize.y << ", " << globalWorkSize.z << "   " << (sliceSmallSize + cut - 1) / cut << endl;

                probeIntersection<<<globalWorkSize, localWorkSize>>>(cudaFillCheck, cudaHashIndex, gridNeighborDim, originGridNeighborDx,
                                                                     gridSESDim, fullSliceGridSESDim, originGridSESDx, cellStartEnd,
                                                                     cudaSortedAtomPosRad, cudaGridValues, /*offset*/ reducedOffset, N, sliceNbCellSES);

                gpuErrchk(cudaPeekAtLastError());
                // No explicit cudaDeviceSynchronize() here: the immediately-following thrust::sort on
                // the default stream is itself synchronizing and ordered after probeIntersection, so
                // the sort already waits for the kernel. Dropping the redundant per-slab full-device
                // sync removes one host stall per slab. (Trades away async-error surfacing at this
                // exact point; errors still surface at the next peek/sync. Verify on dgx.)

                // Count cells at the border, cells that will be used in the refinement step
                thrust::device_ptr<int> fillThrust(cudaFillCheck);
                thrust::sort(fillThrust, fillThrust + sliceNbCellSES);

                unsigned int notEmptyCells = thrust::count_if(thrust::device, fillThrust, fillThrust + sliceNbCellSES, is_notempty());

                if (notEmptyCells == 0)
                {
                    // cerr << "Empty cells !!!" << endl;
                    continue;
                }

                localWorkSize = dim3(NBTHREADS, 1.0f, 1.0f);

                // Too long execution of this kernel triggers the watchdog timer => cut it
                int tranche = min(notEmptyCells, 65536 / 8 * NBTHREADS);

                // streams created once before the slab loop (hoisted)
                int idStream = 0;

                for (unsigned int o = 0; o < notEmptyCells; o += tranche)
                {

                    globalWorkSize = dim3((tranche + NBTHREADS - 1) / NBTHREADS, 1.0f, 1.0f);
                    // cerr <<o<< " Launch (" << globalWorkSize.x << ", "<<globalWorkSize.y<<", "<<globalWorkSize.z<<") x ("<<localWorkSize.x<<", "<<localWorkSize.y<<", 1.0)" << endl;

                    distanceFieldRefine<<<globalWorkSize, localWorkSize, 0, streams[idStream]>>>(cudaFillCheck, cudaHashIndex, gridNeighborDim, originGridNeighborDx,
                                                                                                 gridSESDim, fullSliceGridSESDim, originGridSESDx, cellStartEnd,
                                                                                                 cudaSortedAtomPosRad, cudaGridValues, N, notEmptyCells, reducedOffset, o);

                    idStream++;
                    if (idStream == nbStream)
                        idStream = 0;
                }

                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());

                // streams destroyed once after the slab loop (hoisted)

                // Reset grid values that are outside of the slice

                // localWorkSize = dim3(cut, cut, cut);
                // globalWorkSize = dim3((sliceSize + cut - 1) / cut, (sliceSize + cut - 1) / cut, (sliceSize + cut - 1) / cut);

                // resetGridValuesSlice <<< globalWorkSize, localWorkSize >>> (offset, rangeSearchRefine - 1, fullSliceGridSESDim, cudaGridValues);

                // Marching cubes
                MeshData mesh = computeMarchingCubes(fullSliceGridSESDim, cut, sliceNbCellSES, cudaGridValues,
                                                     vertPerCell, compactedVoxels, gridSESDim, originGridSESDx, reducedOffset,
                                                     cudaSortedAtomPosRad, cellStartEnd, gridNeighborDim, originGridNeighborDx, rangeSearchRefine);

                smoothMeshLaplacian(doSmoothing, mesh);
                resultMeshes.push_back(mesh);

                // View path: stream this completed slab to the consumer immediately, in priority
                // order (visible/nearest first). The mesh.triangles are int3 (per-tri); the callback
                // contract is a flat int array, so emit a temporary flattened copy (dropping the
                // degenerate triangles, matching the consolidated path's filter).
                if (view != NULL && view->cb != NULL)
                {
                    int *flatTris = (int *)malloc(sizeof(int) * mesh.NTriangles * 3);
                    unsigned int nt = 0;
                    for (int t = 0; t < mesh.NTriangles; t++)
                    {
                        int3 tr = mesh.triangles[t];
                        if (tr.x != tr.y && tr.y != tr.z && tr.x != tr.z)
                        {
                            flatTris[nt++] = tr.x;
                            flatTris[nt++] = tr.y;
                            flatTris[nt++] = tr.z;
                        }
                    }
                    view->cb((int)taskId, task.visible ? 1 : 0,
                             mesh.vertices, (unsigned int)mesh.NVertices,
                             flatTris, nt, mesh.atomIdPerVert, view->userData);
                    free(flatTris);
                }

                // if(resultMeshes.size() == 2){
                // return resultMeshes;
                // }
                // break;
            }
        }
    }
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cerr << "Time for step 2 : " << milliseconds << " ms" << std::endl;

    for (int si = 0; si < nbStream; si++)
        cudaStreamDestroy(streams[si]);

    // Device buffers are pooled and kept alive across calls (freed by API_releasePool).

    free(atomPosRad);

#if MEASURETIME
    std::cerr << "Time for computing SES " << (std::clock() - startSES) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
#endif

    return resultMeshes;
}

extern "C"
{
    int NTriangles;
    int NVertices;
    int *globalTriangles;
    float3 *globalVertices;
    int *globalIdAtomPerVert;
}

// Consolidate the per-slab meshes into the global host arrays (globalVertices/Triangles/
// IdAtomPerVert) that API_getVertices/Triangles/AtomIdPerVert return, then free the per-slab
// arrays. Shared by API_computeSES and API_computeSES_view. Sets *NVert/*NTri + the globals.
static void consolidateMeshes(std::vector<MeshData> &resultMeshes, unsigned int *NVert, unsigned int *NTri)
{
    unsigned int totalVerts = 0;
    unsigned int totalTris = 0;

    for (int i = 0; i < resultMeshes.size(); i++)
    {
        totalVerts += resultMeshes[i].NVertices;
        totalTris += resultMeshes[i].NTriangles * 3;
    }
    globalVertices = (float3 *)malloc(sizeof(float3) * totalVerts);
    globalTriangles = (int *)malloc(sizeof(int) * totalTris);
    globalIdAtomPerVert = (int *)malloc(sizeof(int) * totalVerts);

    unsigned int cumulVert = 0;
    unsigned int curIdV = 0;
    unsigned int curIdT = 0;

    for (int i = 0; i < resultMeshes.size(); i++)
    {
        for (int v = 0; v < resultMeshes[i].NVertices; v++)
        {
            globalVertices[curIdV] = resultMeshes[i].vertices[v];
            globalIdAtomPerVert[curIdV] = resultMeshes[i].atomIdPerVert[v];
            curIdV++;
        }
        for (int t = 0; t < resultMeshes[i].NTriangles; t++)
        {
            int3 triangle = resultMeshes[i].triangles[t];
            if (triangle.x != triangle.y && triangle.y != triangle.z && triangle.x != triangle.z)
            {
                globalTriangles[curIdT++] = triangle.x + cumulVert;
                globalTriangles[curIdT++] = triangle.y + cumulVert;
                globalTriangles[curIdT++] = triangle.z + cumulVert;
            }
        }
        cumulVert += resultMeshes[i].NVertices;
    }

    // Free the per-slab MeshData host arrays now they are consolidated into the globals.
    // (Previously leaked: API_freeMesh only frees the consolidated global* arrays.)
    for (int i = 0; i < resultMeshes.size(); i++)
    {
        free(resultMeshes[i].vertices);
        free(resultMeshes[i].triangles);
        free(resultMeshes[i].atomIdPerVert);
    }

    *NVert = totalVerts;
    *NTri = curIdT;
    NTriangles = curIdT;
    NVertices = totalVerts;
}

API void API_computeSES(float resoSES, float3 *atomPos, float *atomRad, unsigned int N, float3 *out_vertices,
                        unsigned int *NVert, int *out_triangles, unsigned int *NTri, int doSmoothing)
{
    *NVert = 0;
    *NTri = 0;

    std::vector<MeshData> resultMeshes = computeSlicedSES(atomPos, atomRad, N, resoSES, doSmoothing);
    consolidateMeshes(resultMeshes, NVert, NTri);
}

API void API_computeSES_view(float resoSES, float3 *atomPos, float *atomRad, unsigned int N,
                             const float *frustumPlanes, float3 camPos, int mode,
                             SlabMeshCallback slabCb, void *userData,
                             unsigned int *NVert, unsigned int *NTri, int doSmoothing)
{
    *NVert = 0;
    *NTri = 0;

    ViewParams view;
    view.enabled  = (frustumPlanes != NULL);
    view.mode     = mode;
    view.camPos   = camPos;
    view.cb       = slabCb;
    view.userData = userData;
    if (frustumPlanes != NULL)
        for (int p = 0; p < 24; p++) view.planes[p] = frustumPlanes[p];

    std::vector<MeshData> resultMeshes = computeSlicedSES(atomPos, atomRad, N, resoSES, doSmoothing, &view);
    // Consolidate the COMPUTED slabs too, so a caller can still fetch one mesh via API_getVertices
    // after the streamed callbacks (in visible-only mode this is the visible subset).
    consolidateMeshes(resultMeshes, NVert, NTri);
}

// Two-band distance LOD (opt-in). Computes the surface in TWO passes with complementary frustums:
//   pass 1 (NEAR): fine voxel `resoSES`, slabs inside `nearPlanes`  -> streamed first (isVisible=1);
//   pass 2 (FAR):  coarse voxel `resoSES*coarseMul`, slabs inside `farPlanes` -> streamed after.
// Each pass is internally uniform (no internal LOD seam); the ONE seam is between the bands, which the
// caller hides by letting nearPlanes and farPlanes OVERLAP slightly (a skirt). Both passes use
// visible-only culling (mode 0), so off-band slabs are skipped. The consolidated API_getVertices mesh
// holds near+far. Cheaper than full-fine because the far band is computed at coarser resolution.
// Validated on dgx (1AON v0.3, 2x coarse far half): ~1.56x vs full-fine while keeping the whole surface.
API void API_computeSES_lod(float resoSES, float coarseMul,
                            float3 *atomPos, float *atomRad, unsigned int N,
                            const float *nearPlanes, const float *farPlanes, float3 camPos,
                            SlabMeshCallback slabCb, void *userData,
                            unsigned int *NVert, unsigned int *NTri, int doSmoothing)
{
    *NVert = 0;
    *NTri = 0;

    // Pass 1 — near band, fine voxel.
    ViewParams vNear;
    vNear.enabled = (nearPlanes != NULL);
    vNear.mode = 0; // visible-only: skip slabs outside the near band
    vNear.camPos = camPos;
    vNear.cb = slabCb;
    vNear.userData = userData;
    if (nearPlanes != NULL) for (int p = 0; p < 24; p++) vNear.planes[p] = nearPlanes[p];
    std::vector<MeshData> nearMeshes = computeSlicedSES(atomPos, atomRad, N, resoSES, doSmoothing, &vNear);

    // Pass 2 — far band, coarse voxel.
    // FAST-SKIP an entirely-empty far band: if the model's (padded) world AABB is fully OUTSIDE the
    // far frustum, no slab can fall in the far band, so the far pass would run a whole computeSlicedSES
    // (grid build + atom thrust::sort + the full slab-enumeration loop) only to produce 0 geometry.
    // The per-slab visible-only cull inside computeSlicedSES skips the heavy refine/MC kernels but NOT
    // this setup+enumeration cost. Test the model AABB against farPlanes with the existing aabbInFrustum,
    // padded CONSERVATIVELY so the test never skips a band that could contribute: the SES grid origin
    // offsets each side by maxAtomRad + probeRadius (see originGridNeighbor) and the grid extent adds
    // 2*maxAtomRad + 4*probeRadius along the longest axis (computeMaxDist), so a per-side pad of
    // maxAtomRad + 2*probeRadius is >= the grid's actual reach — it OVER-covers, never under-covers.
    // Output-identical: when the far AABB is outside, the near pass (which already covered every visible
    // slab) is the whole surface. (Mirrors the UnityMol port's LOD empty-far-pass skip; the per-slab
    // empty-cell early-out CUDA already has, this is the pass-level one.)
    bool runFar = true;
    if (farPlanes != NULL)
    {
        float3 aabbMin, aabbMax; float maxAtomRad;
        getMinMax(atomPos, atomRad, N, &aabbMin, &aabbMax, &maxAtomRad);
        float pad = maxAtomRad + 2.0f * probeRadius; // >= the grid's per-side reach (conservative)
        aabbMin.x -= pad; aabbMin.y -= pad; aabbMin.z -= pad;
        aabbMax.x += pad; aabbMax.y += pad; aabbMax.z += pad;
        runFar = aabbInFrustum(farPlanes, aabbMin, aabbMax);
    }

    std::vector<MeshData> farMeshes;
    if (runFar)
    {
        ViewParams vFar;
        vFar.enabled = (farPlanes != NULL);
        vFar.mode = 0;
        vFar.camPos = camPos;
        vFar.cb = slabCb;
        vFar.userData = userData;
        if (farPlanes != NULL) for (int p = 0; p < 24; p++) vFar.planes[p] = farPlanes[p];
        farMeshes = computeSlicedSES(atomPos, atomRad, N, resoSES * coarseMul, doSmoothing, &vFar);
    }

    // Consolidate near+far into the global arrays for callers that fetch one mesh at the end.
    for (size_t i = 0; i < farMeshes.size(); i++) nearMeshes.push_back(farMeshes[i]);
    consolidateMeshes(nearMeshes, NVert, NTri);
}

extern "C"
{
    API int *API_getTriangles(bool invertTriangles = false)
    {
        if (invertTriangles)
        {
            for (unsigned int t = 0; t < NTriangles / 3; t++)
            {
                int save = globalTriangles[t * 3];
                globalTriangles[t * 3] = globalTriangles[t * 3 + 1];
                globalTriangles[t * 3 + 1] = save;
            }
        }
        return globalTriangles;
    }
    API float3 *API_getVertices()
    {
        return globalVertices;
    }
    API int *API_getAtomIdPerVert()
    {
        return globalIdAtomPerVert;
    }

    API void API_freeMesh()
    {
        free(globalVertices);
        free(globalTriangles);
        free(globalIdAtomPerVert);
    }

    // Release every pooled device buffer. Call on structure unload / app exit.
    // Safe to call repeatedly; the next API_computeSES re-grows the pools as needed.
    API void API_releasePool()
    {
        freePool(poolAtomPosRad);
        freePool(poolSortedAtomPosRad);
        freePool(poolHashIndex);
        freePool(poolCellStartEnd);
        freePool(poolGridValues);
        freePool(poolFillCheck);
        freePool(poolVertPerCell);
        freePool(poolCompactedVoxels);
        freePool(poolVertices);
        freePool(poolVertOri);
        freePool(poolTri);
        freePool(poolAtomIdPerVert);
    }
}

int main(int argc, const char *argv[])
{

    args::ArgumentParser parser("QuickSES, SES mesh generation using GPU", "");
    args::Group groupMandatory(parser, "", args::Group::Validators::All);
    args::Group groupOptional(parser, "", args::Group::Validators::DontCare);
    args::ValueFlag<string> inFile(groupMandatory, "input.pdb", "Input PDB file", {'i'});
    args::ValueFlag<string> outFile(groupMandatory, "output.obj", "Output OBJ mesh file", {'o'});
    args::ValueFlag<int> smoothTimes(groupOptional, "smooth factor", "(1) Times to run Laplacian smoothing step.", {'l'});
    args::ValueFlag<float> voxelSize(groupOptional, "voxel size", "(0.5) Voxel size in Angstrom. Defines the quality of the mesh.", {'v'});
    args::ValueFlag<int> slice(groupOptional, "slice size", "(300) Size of the sub-grid. Defines the quantity of GPU memory needed.", {'s'});
    args::HelpFlag help(groupOptional, "help", "   Display this help menu", {'h', "help"});

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help)
    {
        std::cerr << parser;
        return 0;
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return -1;
    }
    catch (args::ValidationError e)
    {
        // std::cerr << e.what() << std::endl;
        std::cerr << "Usage: " << parser;
        return -1;
    }

    if (inFile)
    {
        inputFilePath = args::get(inFile);
    }
    if (outFile)
    {
        outputFilePath = args::get(outFile);
    }
    if (smoothTimes)
    {
        laplacianSmoothSteps = args::get(smoothTimes);
    }
    if (voxelSize)
    {
        gridResolutionSES = args::get(voxelSize);
    }
    if (slice)
    {
        SLICE = args::get(slice);
    }

    std::clock_t startparse = std::clock();

    initRadiusDic();

    pdb *P;
    P = initPDB();

    parsePDB((char *)inputFilePath.c_str(), P, (char *)"");

    cerr << "Grid resolution = " << gridResolutionSES << endl;
    std::cerr << "Time for parse " << (std::clock() - startparse) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

    unsigned int N = 0;
    std::vector<float3> atomPos;
    std::vector<float> atomRadii;

    atom *A = NULL;
    chain *C = NULL;

    for (int chainId = 0; chainId < P->size; chainId++)
    {
        C = &P->chains[chainId];

        A = &C->residues[0].atoms[0];

        while (A != NULL)
        {
            float3 coords = A->coor;
            atomPos.push_back(coords);
            float atomRad;
            if (radiusDic.count(A->element[0]))
                atomRad = radiusDic[A->element[0]];
            else
                atomRad = radiusDic['X'];
            atomRadii.push_back(atomRad);

            N++;
            A = A->next;
        }
    }

    std::vector<MeshData> resultMeshes = computeSlicedSES(&atomPos[0], &atomRadii[0], N, gridResolutionSES, laplacianSmoothSteps);
    // std::vector<MeshData> resultMeshes = computeSlicedSESCPU(P);

    // Write to OBJ
    writeToObj(outputFilePath, resultMeshes);

    freePDB(P);

    return 0;
}
