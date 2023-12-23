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

#define NBTHREADS 256
#define EMPTYCELL INT_MAX-1
#define PROBERADIUS (1.4f)
#define EPSILON (0.001f)

#include <thrust/binary_search.h>

#include "operators.cu"
#include "MarchingCubes.cu"



struct compare_int2 {
    __host__ __device__ __forceinline__ bool operator()(int2 a, int2 b) {return a.x < b.x;}
};
struct compare_int {
    __host__ __device__ __forceinline__ bool operator()(int a, int b) {return a < b;}
};

struct compare_int3 {
    __host__ __device__ bool operator()(int3 a, int3 b) {
        if      (a.x <= b.x && a.y <= b.y && a.z < b.z) return true;
        else if (a.x <= b.x && a.y < b.y) return true;
        else if (a.x < b.x) return true;
        else return false;
    }
};

struct sameint3 {
    __host__ __device__ bool operator()(int3 a, int3 b) {
        if (a.x != b.x)
            return false;
        if (a.y != b.y)
            return false;
        if (a.z != b.z)
            return false;

        return true;
    }
};

typedef thrust::tuple<float,float, float> vec3;

struct add_uint2 {
    __device__
    uint2 operator()(const uint2& a, const uint2& b) const {
        uint2 r;
        r.x = a.x + b.x;
        r.y = a.y + b.y;
        return r;
    }
};

struct is_notempty
{
    __host__ __device__
    bool operator()(const int &x) const
    {
        return x != EMPTYCELL;
    }
};


__global__ void memsetCudaFloat(float *data, float val, int N){
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (index >= N) {
        return;
    }
    data[index] = val;
}
__global__ void memsetCudaInt(int *data, int val, int N) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (index >= N) {
        return;
    }
    data[index] = val;
}
__global__ void memsetCudaInt2(int2 *data, int2 val, int N) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (index >= N) {
        return;
    }
    data[index] = val;
}
__global__ void memsetCudaUInt2(uint2 *data, uint2 val, int N) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (index >= N) {
        return;
    }
    data[index] = val;
}
__host__ __device__ int3 spaceToGrid(float3 pos3D, float3 originGrid, float dx) {
    float3 tmp = ((pos3D - originGrid) / dx);
    return make_int3(tmp.x, tmp.y, tmp.z);

}
__host__ __device__ float3 gridToSpace(int3 cellPos, float3 originGrid, float dx) {
    return (originGrid + (make_float3(cellPos.x, cellPos.y, cellPos.z) * dx) );
    //return originGrid + (convert_float3(cellPos) * dx) + (dx / 2.0f);
}

__host__ __device__ int flatten3DTo1D(int3 id3d, int3 gridDim) {
    return (gridDim.y * gridDim.z * id3d.x) + (gridDim.z * id3d.y) + id3d.z;
}
// __host__ __device__ int flatten3DTo1D(int3 id3d, int3 gridDim) {
//     // return id3d.x + gridDim.x * (id3d.y + gridDim.z * id3d.z);
//     return id3d.x + id3d.y * gridDim.x + id3d.z * gridDim.x * gridDim.y;
// }


__host__ __device__ int3 unflatten1DTo3D(int index, int3 gridDim) {
    int3 res;
    // res.x = index % gridDim.x;
    // res.y = (index / gridDim.x) % gridDim.y;
    // res.z = index / (gridDim.x * gridDim.y);
    res.x = index / (gridDim.y * gridDim.z); //Note the integer division . This is x
    res.y = (index - res.x * gridDim.y * gridDim.z) / gridDim.z; //This is y
    res.z = index - res.x * gridDim.y * gridDim.z - res.y * gridDim.z; //This is z
    return res;
}

inline __host__ __device__ float fast_distance(float3 p1, float3 p2) {
    float x = (p1.x - p2.x) * (p1.x - p2.x);
    float y = (p1.y - p2.y) * (p1.y - p2.y);
    float z = (p1.z - p2.z) * (p1.z - p2.z);

    return sqrt( x + y + z);
}
inline __host__ __device__ float sqr_distance(float3 p1, float3 p2) {
    float x = (p1.x - p2.x) * (p1.x - p2.x);
    float y = (p1.y - p2.y) * (p1.y - p2.y);
    float z = (p1.z - p2.z) * (p1.z - p2.z);

    return x + y + z;
}

inline __host__ __device__ float3 clamp(float3 f, float a, float b)
{
    return make_float3(max(a, min(f.x, b)),
                       max(a, min(f.y, b)),
                       max(a, min(f.z, b))
                      );
}
inline __host__ __device__ int3 clamp(int3 f, int a, int b)
{
    return make_int3(max(a, min(f.x, b)),
                     max(a, min(f.y, b)),
                     max(a, min(f.z, b))
                    );
}
inline __host__ __device__ int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}


// calculate cell address as the hash value for each atom
__global__ void hashAtoms(unsigned int natoms,
                          float4*  xyzr,
                          int3 gridDim,
                          float4 originGridDDx,
                          int2 * atomHashIndex,
                          unsigned natomsnextPow2) {


    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= natomsnextPow2)
        return;

    if (index >= natoms) {
        atomHashIndex[index].x = INT_MAX;
        atomHashIndex[index].y = INT_MAX;
        return;
    }

    float4 atom = xyzr[index];// read atom coordinate and radius
    float3 pos = make_float3(atom.x, atom.y, atom.z);
    // compute cell index, clamped to fall within grid bounds
    float3 originGrid = make_float3(originGridDDx.x, originGridDDx.y, originGridDDx.z);
    float dx = originGridDDx.w;
    int3 cell = spaceToGrid(pos, originGrid, dx);

    int hash = flatten3DTo1D(cell, gridDim);

    atomHashIndex[index].x = hash;// atoms hashed to cell address
    atomHashIndex[index].y = index; // original atom index

}


//https://github.com/FROL256/opencl_bitonic_sort_by_key/blob/master/bitonic_sort_gpu.h

inline __device__ int getKey(int2 v) { return v.x; }
inline __device__ int getVal(int2 v) { return v.y; }

inline __device__ bool compare(int2 a, int2 b) { return getKey(a) < getKey(b); }

__global__ void bitonic_pass_kernel(int2* theArray, int stage, int passOfStage, int a_invertModeOn)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    const int r     = 1 << (passOfStage);
    const int lmask = r - 1;

    const int left  = ((j >> passOfStage) << (passOfStage + 1)) + (j & lmask);
    const int right = left + r;

    const int2 a = theArray[left];
    const int2 b = theArray[right];

    const bool cmpRes = compare(a, b);

    const int2 minElem = cmpRes ? a : b;
    const int2 maxElem = cmpRes ? b : a;

    const int oddEven = j >> stage;

    const bool isSwap = (oddEven & 1) & a_invertModeOn;

    const int minId = isSwap ? right : left;
    const int maxId = isSwap ? left  : right;

    theArray[minId] = minElem;
    theArray[maxId] = maxElem;
}


__global__ void bitonic_512( int2* theArray, int stage, int passOfStageBegin, int a_invertModeOn)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int lid = threadIdx.x;

    int blockId = (tid / 256);

    __shared__ int2 s_array[512];

    s_array[lid + 0]   = theArray[blockId * 512 + lid + 0];
    s_array[lid + 256] = theArray[blockId * 512 + lid + 256];

    __syncthreads();

    for (int passOfStage = passOfStageBegin; passOfStage >= 0; passOfStage--)
    {
        const int j     = lid;
        const int r     = 1 << (passOfStage);
        const int lmask = r - 1;

        const int left  = ((j >> passOfStage) << (passOfStage + 1)) + (j & lmask);
        const int right = left + r;

        const int2 a = s_array[left];
        const int2 b = s_array[right];

        const bool cmpRes = compare(a, b);

        const int2 minElem = cmpRes ? a : b;
        const int2 maxElem = cmpRes ? b : a;

        const int oddEven = tid >> stage; // (j >> stage)

        const bool isSwap = (oddEven & 1) & a_invertModeOn;

        const int minId = isSwap ? right : left;
        const int maxId = isSwap ? left  : right;

        s_array[minId] = minElem;
        s_array[maxId] = maxElem;

        __syncthreads();
    }

    theArray[blockId * 512 + lid + 0]   = s_array[lid + 0];
    theArray[blockId * 512 + lid + 256] = s_array[lid + 256];

}


__global__ void sortCell(unsigned int natoms, float4 *xyzr, int2 *atomHashIndex,
                         float4 *sorted_xyzr,
                         int2 *cellStartEnd) {

    int hash;
    int id;
    // __local int local_hash[NBTHREADS + 1];
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= natoms)
        return;

    hash = atomHashIndex[index].x;
    id = atomHashIndex[index].y;

    int hashm1;
    if (index != 0)
        hashm1 = atomHashIndex[index - 1].x;
    else
        hashm1 = hash;


    if (index == 0 || hash != hashm1) {
        cellStartEnd[hash].x = index; // set start
        if (index > 0)
            cellStartEnd[hashm1].y = index; // set end
    }

    if (index == natoms - 1) {
        cellStartEnd[hash].y = index + 1; // set end
    }

    // Reorder atoms according to sorted indices
    sorted_xyzr[index] = xyzr[id];

}


inline __host__ __device__ float computeInOrOut(int3 id3DNeigh, int2 *cellStartEnd, float4 *sorted_xyzr, float3 spacePosSES, float dx, int3 gridDimNeighbor) {
    //Loop over neighbors to compute if the cell is:
    // 1) - ouside => result = PROBERADIUS
    // 2) - inside => result = -dx
    // 3) - at the border =>  result = 0.0
    int3 curgridId;
    float result = PROBERADIUS;
    bool nearProbe = false;

// #pragma unroll
    for (int x = -1; x <= 1; x++) {
        curgridId.x = clamp(id3DNeigh.x + x, 0, gridDimNeighbor.x - 1);
// #pragma unroll
        for (int y = -1; y <= 1; y++) {
            curgridId.y = clamp(id3DNeigh.y + y, 0, gridDimNeighbor.x - 1);
// #pragma unroll
            for (int z = -1; z <= 1; z++) {
                curgridId.z = clamp(id3DNeigh.z + z, 0, gridDimNeighbor.x - 1);

                int neighcellhash = flatten3DTo1D(curgridId, gridDimNeighbor);
                int idStart = cellStartEnd[neighcellhash].x;
                int idStop = cellStartEnd[neighcellhash].y;

                if (idStart < EMPTYCELL) {
                    for (int id = idStart; id < idStop; id++) {
                        float4 xyzr = sorted_xyzr[id];
                        float rad = xyzr.w;
                        float3 pos = make_float3(xyzr.x, xyzr.y, xyzr.z);
                        // float d = sqr_distance(pos, spacePosSES);
                        float d = fast_distance(pos, spacePosSES);

                        // minDist = min(d, minDist);
                        // if (d < rad * rad)
                        if (d < rad - dx)
                            return -dx;
                        // if (d < (PROBERADIUS + rad) * (PROBERADIUS + rad))
                        if (d < (PROBERADIUS + rad) )
                            nearProbe = true;
                    }
                }
            }
        }
    }    

    //float result = (1 - nearProbe) * PROBERADIUS;
    if (nearProbe)
        result = 0.0f;
    return result;
}

__global__ void probeIntersection(int * checkFill,
                                  int2 * atomHashIndex,
                                  int3 gridDimNeighbor,
                                  float4 originGridNeighborDDx,
                                  int3 gridDimSES,
                                  int3 sliceGridDimSES,
                                  // int3 smallsliceGridDimSES,
                                  float4 originGridSESDx,
                                  int2 *cellStartEnd,
                                  float4 * sorted_xyzr,
                                  float *gridValues,
                                  int3 offsetGrid,
                                  unsigned int N,
                                  int sliceNb) {

    // Get global position in X direction
    unsigned int i = (threadIdx.x + blockIdx.x * blockDim.x);
    // Get global position in Y direction
    unsigned int j = (threadIdx.y + blockIdx.y * blockDim.y);
    // Get global position in Z direction
    unsigned int k = (threadIdx.z + blockIdx.z * blockDim.z);

    int3 ijk = make_int3(i, j, k);

    if (i >= sliceGridDimSES.x - 1)
        return;
    if (j >= sliceGridDimSES.y - 1)
        return;
    if (k >= sliceGridDimSES.z - 1)
        return;



    int hash = flatten3DTo1D(ijk, sliceGridDimSES);

    // if(hash >= sliceNb)
    //     return;


    int3 ijkOffset = make_int3(i + offsetGrid.x, j + offsetGrid.y, k + offsetGrid.z);
    int hashOffset = flatten3DTo1D(ijkOffset, gridDimSES);

    if (ijkOffset.x >= gridDimSES.x - 1)
        return;
    if (ijkOffset.y >= gridDimSES.y - 1)
        return;
    if (ijkOffset.z >= gridDimSES.z - 1)
        return;


    float3 originGridNeighbor = make_float3(originGridNeighborDDx.x, originGridNeighborDDx.y, originGridNeighborDDx.z);
    float dxNeighbor = originGridNeighborDDx.w;
    float dxSES = originGridSESDx.w;

    float3 spacePos3DCellSES = gridToSpace(ijkOffset, originGridNeighbor, dxSES);

    // id of the current cell in the neighbor grid
    int3 gridPos3DCellNeighbor = spaceToGrid(spacePos3DCellSES, originGridNeighbor, dxNeighbor);


    //Loop over neighbors to compute if the cell is:
    // 1) - ouside => result = PROBERADIUS
    // 2) - inside => result = -dx
    // 3) - at the border =>  result = 0.0
    float result = computeInOrOut(gridPos3DCellNeighbor, cellStartEnd, sorted_xyzr, spacePos3DCellSES, dxSES, gridDimNeighbor);


    //If should process => record id, if not record a large number
    int fill = EMPTYCELL;

    int range = (int)ceil(PROBERADIUS / dxSES);
    if (abs(result) < EPSILON){
        fill = hash;
        // if(i < range || j < range || k < range || i > smallsliceGridDimSES.x + range || j > smallsliceGridDimSES.y + range || k > smallsliceGridDimSES.z + range){
        //     fill = EMPTYCELL;
        // }
    }

    checkFill[hash] = fill;
    gridValues[hash] = result;

}


__global__ void distanceFieldRefine(int * checkFill, int2 * atomHashIndex, int3 gridDimNeighbor,
                                    float4 originGridNeighborDDx, int3 gridDimSES, int3 sliceGridDimSES,
                                    float4 originGridSESDx, int2 * cellStartEnd, float4 * sorted_xyzr,
                                    float *gridValues, unsigned int totalCells, unsigned int notEmptyCells,
                                    int3 offsetGrid, unsigned int of) {

    int id = (threadIdx.x + blockIdx.x * blockDim.x) + of;


    if (id >= notEmptyCells) {
        return;
    }

    int hash = checkFill[id];

    int3 ijk = unflatten1DTo3D(hash, sliceGridDimSES);

    //From slice to real id in the full size grid
    int3 ijkOffset = make_int3(ijk.x + offsetGrid.x, ijk.y + offsetGrid.y, ijk.z + offsetGrid.z);
    int hashOffset = flatten3DTo1D(ijkOffset, gridDimSES);

    float3 originGridSES = make_float3(originGridSESDx.x, originGridSESDx.y, originGridSESDx.z);

    float dxSES = originGridSESDx.w;

    float3 spacePos3DCellSES = gridToSpace(ijkOffset, originGridSES, dxSES);

    float newresult = -dxSES;

    const int idSESRangeToSearch = (int)ceil(PROBERADIUS / dxSES);
    const float pme = PROBERADIUS - EPSILON;
    float minDist = 100000.0f;

    int3 curgridSESId;

    //Find the closest outside SES cell in the range [-probeRadius, +probeRadius]
// #pragma unroll
    for (int x = -idSESRangeToSearch; x <= idSESRangeToSearch; x++) {

        curgridSESId.x = clamp(ijk.x + x , 0, sliceGridDimSES.x - 1);
// #pragma unroll
        for (int y = -idSESRangeToSearch; y <= idSESRangeToSearch; y++) {
            curgridSESId.y = clamp(ijk.y + y , 0, sliceGridDimSES.y - 1);
// #pragma unroll
            for (int z = -idSESRangeToSearch; z <= idSESRangeToSearch; z++) {
                curgridSESId.z = clamp(ijk.z + z , 0, sliceGridDimSES.z - 1);
                int curgrid1DSESId = flatten3DTo1D(curgridSESId, sliceGridDimSES);

                if (gridValues[curgrid1DSESId] > pme) {//Outside
                    
                    int3 curgrid3DSESIdOffset = make_int3(curgridSESId.x + offsetGrid.x,
                                                        curgridSESId.y + offsetGrid.y,
                                                        curgridSESId.z + offsetGrid.z);

                    float3 spacePosSES = gridToSpace(curgrid3DSESIdOffset, originGridSES, dxSES);
                    //Distance from our current grid cell to the outside grid cell
                    float d = fast_distance(spacePosSES, spacePos3DCellSES);
                    minDist = min(d, minDist);
                }
            }
        }
    }
    if (minDist < 999.0f)
        newresult =  PROBERADIUS - minDist;

    gridValues[hash] = newresult;

}

inline __host__ __device__ int computeClosestAtom(float3 vert, int3 id3DNeigh, int2 *cellStartEnd, float4 *sorted_xyzr, int3 gridDimNeighbor) {


    int closestId = -1;
    float minD = 999999.0f;
    int3 curgridId;
// #pragma unroll
    for (int x = -1; x <= 1; x++) {
        curgridId.x = clamp(id3DNeigh.x + x, 0, gridDimNeighbor.x - 1);
// #pragma unroll
        for (int y = -1; y <= 1; y++) {
            curgridId.y = clamp(id3DNeigh.y + y, 0, gridDimNeighbor.x - 1);
// #pragma unroll
            for (int z = -1; z <= 1; z++) {
                curgridId.z = clamp(id3DNeigh.z + z, 0, gridDimNeighbor.x - 1);

                int neighcellhash = flatten3DTo1D(curgridId, gridDimNeighbor);
                int idStart = cellStartEnd[neighcellhash].x;
                int idStop = cellStartEnd[neighcellhash].y;

                if (idStart < EMPTYCELL) {
                    for (int id = idStart; id < idStop; id++) {
                        float4 xyzr = sorted_xyzr[id];
                        float3 pos = make_float3(xyzr.x, xyzr.y, xyzr.z);
                        // float d = sqr_distance(pos, vert);
                        float d = fast_distance(pos, vert);
                        if(d < minD){
                            minD = d;
                            closestId = id;
                        }
                    }
                }
            }
        }
    }
    return closestId;


}

__global__ void closestAtomPerVertex(int *atomIdPerVert, float3 *vertices, unsigned int Nvert, int3 gridDimNeighbor,
                                    float4 originGridNeighborDDx,
                                    float4 originGridSESDx, int2 * cellStartEnd, float4 * sorted_xyzr) {

    int id = (threadIdx.x + blockIdx.x * blockDim.x);

    if(id >= Nvert)
        return;

    float3 vert = vertices[id];

    float3 originGridNeighbor = make_float3(originGridNeighborDDx.x, originGridNeighborDDx.y, originGridNeighborDDx.z);
    float dxNeighbor = originGridNeighborDDx.w;

    int3 gridPos3DCellNeighbor = spaceToGrid(vert, originGridNeighbor, dxNeighbor);

    atomIdPerVert[id] = computeClosestAtom(vert, gridPos3DCellNeighbor, cellStartEnd, sorted_xyzr, gridDimNeighbor);
}

__global__ void resetGridValuesSlice(const int3 offset, const int rangeSearchRefine, const int3 sliceGridDimSES, float *gridValues){

    // Get global position in X direction
    unsigned int i = (threadIdx.x + blockIdx.x * blockDim.x);
    // Get global position in Y direction
    unsigned int j = (threadIdx.y + blockIdx.y * blockDim.y);
    // Get global position in Z direction
    unsigned int k = (threadIdx.z + blockIdx.z * blockDim.z);

    int3 ijk = make_int3(i, j, k);

    if (i >= sliceGridDimSES.x - 1)
        return;
    if (j >= sliceGridDimSES.y - 1)
        return;
    if (k >= sliceGridDimSES.z - 1)
        return;


    int hash = flatten3DTo1D(ijk, sliceGridDimSES);

    if(i >= sliceGridDimSES.x - rangeSearchRefine)
        gridValues[hash] = PROBERADIUS;
    if(j >= sliceGridDimSES.y - rangeSearchRefine)
        gridValues[hash] = PROBERADIUS;
    if(k >= sliceGridDimSES.z - rangeSearchRefine)
        gridValues[hash] = PROBERADIUS;

    if(offset.x != 0){
        if(i < rangeSearchRefine)
            gridValues[hash] = PROBERADIUS;
    }
    else{
        if(i >= sliceGridDimSES.x - rangeSearchRefine * 2)
            gridValues[hash] = PROBERADIUS;
    }
    if(offset.y != 0){
        if(j < rangeSearchRefine)
            gridValues[hash] = PROBERADIUS;
    }
    else{
        if(j >= sliceGridDimSES.y - rangeSearchRefine * 2)
            gridValues[hash] = PROBERADIUS;
    }
    if(offset.z != 0){
        if(k < rangeSearchRefine)
            gridValues[hash] = PROBERADIUS;
    }
    else{
        if(k >= sliceGridDimSES.z - rangeSearchRefine * 2)
            gridValues[hash] = PROBERADIUS;
    }


}