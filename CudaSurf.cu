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

//#include <cassert>
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
#include <thrust/sequence.h>
#include <thrust/binary_search.h>

using namespace std;


int SLICE = 300;
float probeRadius = 1.4f;
float gridResolutionNeighbor;
float gridResolutionSES = 0.5f;
int laplacianSmoothSteps = 1;
string outputFilePath = "output.obj";
string inputFilePath = "";



unsigned int getMinMax(chain *C, float3 *minVal, float3 *maxVal, float *maxAtom) {
    atom *A = NULL;
    unsigned int N = 0;


    A = &C->residues[0].atoms[0];
    float3 vmin, vmax, coords;

    vmin.x = vmin.y = vmin.z = 100000.0f;
    vmax.x = vmax.y = vmax.z = -100000.0f;
    *maxAtom = 0.0f;
    while (A != NULL) {
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
unsigned int getMinMax(pdb *P, float3 *minVal, float3 *maxVal, float *maxAtom) {
    atom *A = NULL;
    unsigned int N = 0;
    chain *C = NULL;
    *maxAtom = 0.0f;
    float3 vmin, vmax, coords;

    vmin.x = vmin.y = vmin.z = 100000.0f;
    vmax.x = vmax.y = vmax.z = -100000.0f;

    for (int chainId = 0; chainId < P->size; chainId++) {
        C = &P->chains[chainId];

        A = &C->residues[0].atoms[0];

        while (A != NULL) {
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
void getMinMax(float3 *positions, float *radii, unsigned int N, float3 *minVal, float3 *maxVal, float *maxAtom) {
    *maxAtom = 0.0f;
    float3 vmin, vmax, coords;

    vmin.x = vmin.y = vmin.z = 100000.0f;
    vmax.x = vmax.y = vmax.z = -100000.0f;

    for (unsigned int a = 0; a < N; a++) {
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

float4 *getArrayAtomPosRad(chain *C, unsigned int N) {

    float4 *result = new float4[N];
    atom *A = NULL;
    int id = 0;

    A = &C->residues[0].atoms[0];
    float3 coords;
    while (A != NULL) {
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


float4 *getArrayAtomPosRad(pdb *P, unsigned int N) {
    chain *C = NULL;
    atom *A = NULL;
    float4 *result = new float4[N];
    // float4 *result;
    // cudaMallocHost((void **)&result, sizeof(float4) * N);
    int id = 0;

    for (int chainId = 0; chainId < P->size; chainId++) {
        C = &P->chains[chainId];

        A = &C->residues[0].atoms[0];
        float3 coords;
        while (A != NULL) {
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

float4 *getArrayAtomPosRad(float3 *positions, float *radii, unsigned int N) {
    float4 *result = (float4 *)malloc(sizeof(float4) * N);
    int id = 0;

    for (int a = 0; a < N; a++) {
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


float computeMaxDist(float3 minVal, float3 maxVal, float maxAtomRad) {
    return std::max(maxVal.x - minVal.x, std::max(maxVal.y - minVal.y, maxVal.z - minVal.z)) + (2 * maxAtomRad) + (4 * probeRadius);
}


/*std::vector<MeshData> computeSlicedSESCPU(pdb *&P) {

    //Record a mesh per slice
    std::vector<MeshData> resultMeshes;

    float3 minVal, maxVal;
    float maxAtomRad = 0.0;
    unsigned int N = getMinMax(P, &minVal, &maxVal, &maxAtomRad);

    if (N <= 1) {
        cerr << "Failed to parse the PDB or empty PDB file" << endl;
        return resultMeshes;
    }

    float4 *atomPosRad = getArrayAtomPosRad(P, N);
    float maxDist = computeMaxDist(minVal, maxVal, maxAtomRad);

    gridResolutionNeighbor = probeRadius + maxAtomRad;

    //Grid is a cube
    float3 originGridNeighbor = {
        minVal.x - maxAtomRad - 2 * probeRadius,
        minVal.y - maxAtomRad - 2 * probeRadius,
        minVal.z - maxAtomRad - 2 * probeRadius
    };

    int gridNeighborSize = (int)ceil(maxDist / gridResolutionNeighbor);

    int3 gridNeighborDim = {gridNeighborSize, gridNeighborSize, gridNeighborSize};

    int gridSESSize = (int)ceil(maxDist / gridResolutionSES);

    int3 gridSESDim = {gridSESSize, gridSESSize, gridSESSize};

    float4 originGridNeighborDx = {
        originGridNeighbor.x,
        originGridNeighbor.y,
        originGridNeighbor.z,
        gridResolutionNeighbor
    };

    float4 originGridSESDx = {
        originGridNeighborDx.x,
        originGridNeighborDx.y,
        originGridNeighborDx.z,
        gridResolutionSES
    };

    unsigned int nbcellsNeighbor = gridNeighborDim.x * gridNeighborDim.y * gridNeighborDim.z;
    // unsigned int nbcellsSES = gridSESDim.x * gridSESDim.y * gridSESDim.z;


    float4 *cudaSortedAtomPosRad;
    int2 *cudaHashIndex;
    int2 *cellStartEnd;
    float *cudaGridValues;
    int *cudaFillCheck;

    //Marching cubes data
    uint2* vertPerCell;
    unsigned int *compactedVoxels;


    cudaSortedAtomPosRad = (float4 *)malloc(sizeof(float4) * N);
    cudaHashIndex = (int2 *)malloc(sizeof(int2) * N);
    cellStartEnd = (int2 *)malloc(sizeof(int2) * nbcellsNeighbor);


    //-------------- Step 1 : Insert atoms in neighbor cells -----------------

    //hashAtoms
    for (int i = 0; i < N; i++) {
        int3 cell = spaceToGrid(make_float3(atomPosRad[i].x, atomPosRad[i].y, atomPosRad[i].z), originGridNeighbor, gridResolutionNeighbor);
        int hash = flatten3DTo1D(cell, gridNeighborDim);
        cudaHashIndex[i] = make_int2(hash, i);
    }

    std::vector<int2> hashindex(cudaHashIndex, cudaHashIndex + N);
    std::sort(hashindex.begin(), hashindex.end(), compare_int2());
    cudaHashIndex = hashindex.data();

    for (int i = 0; i < nbcellsNeighbor; i++) {
        cellStartEnd[i].x = EMPTYCELL;
        cellStartEnd[i].y = EMPTYCELL;
    }

    //SortCell
    for (int i = 0; i < N; i++) {
        int hash = cudaHashIndex[i].x;
        int id = cudaHashIndex[i].y;


        int hashm1;
        if (i != 0)
            hashm1 = cudaHashIndex[i - 1].x;
        else
            hashm1 = hash;


        if (i == 0 || hash != hashm1) {
            cellStartEnd[hash].x = i; // set start
            if (i > 0)
                cellStartEnd[hashm1].y = i; // set end
        }

        if (i == N - 1) {
            cellStartEnd[hash].y = i + 1; // set end
        }

        // Reorder atoms according to sorted indices
        cudaSortedAtomPosRad[i] = atomPosRad[id];
    }


    // //-------------- Step 2 : Compute points of the grid outside or inside the surface -----------------
    // //Use slices of the grid to avoid allocating large amount of data
    int rangeSearchRefine = (int)ceil(PROBERADIUS / gridResolutionSES);
    int sliceSmallSize = min(SLICE , gridSESSize);
    int sliceSize = min(SLICE + 2 * rangeSearchRefine, gridSESSize);
    int sliceSmallNbCellSES = sliceSmallSize * sliceSmallSize * sliceSmallSize;
    int sliceNbCellSES = sliceSize * sliceSize * sliceSize;
    int3 sliceGridSESDim = make_int3(sliceSmallSize, sliceSmallSize, sliceSmallSize);
    int3 fullSliceGridSESDim = make_int3(sliceSize, sliceSize, sliceSize);

    cudaGridValues = (float *)malloc(sizeof(float) * sliceNbCellSES);
    cudaFillCheck = (int *)malloc(sizeof(int) * sliceNbCellSES);

    vertPerCell = (uint2 *)malloc(sizeof(uint2) * sliceNbCellSES);
    compactedVoxels = (unsigned int *)malloc(sizeof(unsigned int) * sliceNbCellSES);

    cerr << "Allocating " << (sizeof(float) * sliceNbCellSES + 3 * sizeof(int) * sliceNbCellSES) / 1000000.0f << " Mo" << endl;

    int3 offset = {0, 0, 0};
    int cut = 8;

    // cerr << "Full size grid = " << gridSESSize << " x " << gridSESSize << " x " << gridSESSize << endl;
    // // cudaEventRecord(start);
    // // for (int slice = 0; slice < gridSESSize; slice += sliceSmallSize) {
    for (int i = 0; i < gridSESSize; i += sliceSmallSize) {
        offset.x = i;
        for (int j = 0; j < gridSESSize; j += sliceSmallSize) {
            offset.y = j;
            for (int k = 0; k < gridSESSize; k += sliceSmallSize) {
                offset.z = k;

                for (int a = 0; a < sliceNbCellSES; a++) {
                    cudaGridValues[a] = probeRadius;
                    cudaFillCheck[a] = EMPTYCELL;
                }
                int3 reducedOffset = make_int3(max(0, offset.x - rangeSearchRefine),
                                               max(0, offset.y - rangeSearchRefine),
                                               max(0, offset.z - rangeSearchRefine));

                cerr << " ============== " << "offset = " << offset.x << "/" << offset.y << "/" << offset.z << " |  slice size = " << fullSliceGridSESDim.x << " ============== " << endl;
                for (int x = 0; x < sliceSize; x++) {
                    for (int y = 0; y < sliceSize; y++) {
                        for (int z = 0; z < sliceSize; z++) {
                            int3 ijk = make_int3(x, y, z);
                            // unsigned int hash = flatten3DTo1D(ijk, sliceGridSESDim);
                            unsigned int hash = flatten3DTo1D(ijk, fullSliceGridSESDim);


                            if (x >= sliceSize - 1 || y >= sliceSize - 1 || z >= sliceSize - 1)
                                continue;

                            int3 ijkOffset = make_int3(x + reducedOffset.x, y + reducedOffset.y, z + reducedOffset.z);
                            if (ijkOffset.x >= gridSESSize - 1 || ijkOffset.y >= gridSESSize - 1 || ijkOffset.z >= gridSESSize - 1) {
                                continue;
                            }
                            unsigned int hashOffset = flatten3DTo1D(ijkOffset, gridSESDim);
                            float3 spacePos3DCellSES = gridToSpace(ijkOffset, originGridNeighbor, gridResolutionSES);

                            // id of the current cell in the neighbor grid
                            int3 gridPos3DCellNeighbor = spaceToGrid(spacePos3DCellSES, originGridNeighbor, gridResolutionNeighbor);

                            float result = computeInOrOut(gridPos3DCellNeighbor, cellStartEnd, cudaSortedAtomPosRad, spacePos3DCellSES, gridResolutionSES, gridNeighborDim);

                            int fill = EMPTYCELL;

                            if (abs(result) < EPSILON) {
                                fill = hash;

                                if (x < rangeSearchRefine || y < rangeSearchRefine || z < rangeSearchRefine ||
                                        x > sliceSmallSize + rangeSearchRefine || y > sliceSmallSize + rangeSearchRefine || z > sliceSmallSize + rangeSearchRefine) {
                                    fill = EMPTYCELL;
                                }
                            }

                            cudaFillCheck[hash] = fill;
                            cudaGridValues[hash] = result;

                        }
                    }
                }

                std::vector<int> fillvec(cudaFillCheck, cudaFillCheck + sliceNbCellSES);
                std::sort(fillvec.begin(), fillvec.end());
                cudaFillCheck = fillvec.data();
                int notEmptyCell = 0;
                for (int a = 0; a < sliceNbCellSES; a++) {
                    if (cudaFillCheck[a] == EMPTYCELL) {
                        break;
                    }
                    notEmptyCell++;
                }
                cerr << "Not empty = " << notEmptyCell << endl;

                //--------Distance refinement

                // for(int ne = 0; ne < notEmptyCell; ne++){

                //     int hashNe = cudaFillCheck[ne];
                //     int3 ijk = unflatten1DTo3D(hashNe, fullSliceGridSESDim);

                //     const int idSESRangeToSearch = (int)ceil(PROBERADIUS / gridResolutionSES);
                //     const float pme = PROBERADIUS - EPSILON;
                //     float minDist = 100000.0f;
                //     float newresult = -gridResolutionSES;

                //     int3 ijkOffset = make_int3(ijk.x + reducedOffset.x, ijk.y + reducedOffset.y, ijk.z + reducedOffset.z);
                //     float3 spacePos3DCellSES = gridToSpace(ijkOffset, originGridNeighbor, gridResolutionSES);


                //     int3 curgridSESId;

                //         //Find the closest outside SES cell in the range [-probeRadius, +probeRadius]
                //     // #pragma unroll
                //         for (int x = -idSESRangeToSearch; x <= idSESRangeToSearch; x++) {

                //             curgridSESId.x = clamp(ijk.x + x , 0, fullSliceGridSESDim.x - 1);
                //     // #pragma unroll
                //             for (int y = -idSESRangeToSearch; y <= idSESRangeToSearch; y++) {
                //                 curgridSESId.y = clamp(ijk.y + y , 0, fullSliceGridSESDim.y - 1);
                //     // #pragma unroll
                //                 for (int z = -idSESRangeToSearch; z <= idSESRangeToSearch; z++) {
                //                     curgridSESId.z = clamp(ijk.z + z , 0, fullSliceGridSESDim.z - 1);
                //                     int curgrid1DSESId = flatten3DTo1D(curgridSESId, fullSliceGridSESDim);

                //                     if (cudaGridValues[curgrid1DSESId] > pme) {//Outside

                //                         int3 curgrid3DSESIdOffset = make_int3(curgridSESId.x + reducedOffset.x,
                //                                                             curgridSESId.y + reducedOffset.y,
                //                                                             curgridSESId.z + reducedOffset.z);

                //                         float3 spacePosSES = gridToSpace(curgrid3DSESIdOffset, originGridNeighbor, gridResolutionSES);
                //                         //Distance from our current grid cell to the outside grid cell
                //                         float d = fast_distance(spacePosSES, spacePos3DCellSES);
                //                         minDist = min(d, minDist);
                //                     }
                //                 }
                //             }
                //         }
                //         if (minDist < 999.0f)
                //             newresult =  PROBERADIUS - minDist;

                //         cudaGridValues[hashNe] = newresult;
                // }


                if (i == 0 && j == 0 && k == 0) {
                    FILE *fptr;
                    string filename = "debug.dx";
                    if ((fptr = fopen(filename.c_str(), "w")) != NULL) {
                        fprintf(fptr, "object 1 class gridpositions counts %d %d %d\n", sliceSize, sliceSize, sliceSize);
                        fprintf(fptr, "origin %f %f %f\n", originGridSESDx.x, originGridSESDx.y, originGridSESDx.z );
                        fprintf(fptr, "delta %f %f %f\n", originGridSESDx.w, 0.0f, 0.0f);
                        fprintf(fptr, "delta %f %f %f\n", 0.0f, originGridSESDx.w, 0.0f);
                        fprintf(fptr, "delta %f %f %f\n", 0.0f, 0.0f, originGridSESDx.w);
                        fprintf(fptr, "object 2 class gridconnections counts %d %d %d\n", sliceSize, sliceSize, sliceSize);
                        fprintf(fptr, "object 3 class array type double rank 0 items %d data follows\n", sliceNbCellSES);
                        for (int a = 0; a < sliceNbCellSES; a++) {
                            if (a != 0 && a % 3 == 0) {
                                fprintf(fptr, "\n");
                            }
                            fprintf(fptr, "%f ", cudaGridValues[a]);
                        }
                        fprintf(fptr, "\nattribute \"dep\" string \"positions\"\nobject \"regular positions regular connections\" class field\ncomponent \"positions\" value 1\ncomponent \"connections\" value 2\ncomponent \"data\" value 3\n");

                        fclose(fptr);
                    }
                }


                break;
            }
            break;
        }
        break;
    }

    return resultMeshes;
}*/

void computePocketVolume(float3 positions[], float radii[], unsigned int N, float resoSES){

    float3 minVal, maxVal;
    float maxAtomRad = 0.0;

    getMinMax(positions, radii, N, &minVal, &maxVal, &maxAtomRad);

    if (N <= 1) {
        cerr << "Failed to parse the PDB or empty PDB file" << endl;
        return;
    }

    float4 *atomPosRad = getArrayAtomPosRad(positions, radii, N);
    float maxDist = computeMaxDist(minVal, maxVal, maxAtomRad);

    gridResolutionNeighbor = probeRadius + maxAtomRad;

    //Grid is a cube
    float3 originGridNeighbor = {
        minVal.x - maxAtomRad - 2 * probeRadius,
        minVal.y - maxAtomRad - 2 * probeRadius,
        minVal.z - maxAtomRad - 2 * probeRadius
    };

    int gridNeighborSize = (int)ceil(maxDist / gridResolutionNeighbor);

    int3 gridNeighborDim = {gridNeighborSize, gridNeighborSize, gridNeighborSize};

    int gridSESSize = (int)ceil(maxDist / resoSES);

    int3 gridSESDim = {gridSESSize, gridSESSize, gridSESSize};

    float4 originGridNeighborDx = {
        originGridNeighbor.x,
        originGridNeighbor.y,
        originGridNeighbor.z,
        gridResolutionNeighbor
    };

    float4 originGridSESDx = {
        originGridNeighborDx.x,
        originGridNeighborDx.y,
        originGridNeighborDx.z,
        resoSES
    };

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

    //Marching cubes data
    uint2* vertPerCell;
    unsigned int *compactedVoxels;



    gpuErrchk(cudaMalloc((void **)&cudaAtomPosRad , sizeof(float4) * N));
    gpuErrchk(cudaMalloc((void **)&cudaSortedAtomPosRad , sizeof(float4) * N));
    gpuErrchk(cudaMalloc((void **)&cudaHashIndex, sizeof(int2) * N));
    gpuErrchk(cudaMalloc((void**)&cellStartEnd, sizeof(int2) * nbcellsNeighbor));

    //-------------- Step 1 : Insert atoms in neighbor cells -----------------

    //Copy atom positions and radii to GPU
    gpuErrchk(cudaMemcpy(cudaAtomPosRad, atomPosRad, sizeof(float4) * N, cudaMemcpyHostToDevice));

    //Compute atom cell ids
    hashAtoms <<< N, NBTHREADS >>>(N, cudaAtomPosRad, gridNeighborDim, originGridNeighborDx, cudaHashIndex, N);

    gpuErrchk( cudaPeekAtLastError() );

    //Sort atoms cell id
    compare_int2 cmp;
    thrust::device_ptr<int2> D_beg = thrust::device_pointer_cast(cudaHashIndex);
    thrust::sort(D_beg, D_beg + N, cmp);
    gpuErrchk( cudaPeekAtLastError() );

    memsetCudaInt2 <<< (nbcellsNeighbor + NBTHREADS - 1) / NBTHREADS, NBTHREADS >>> (cellStartEnd, make_int2(EMPTYCELL, EMPTYCELL), nbcellsNeighbor);

    //Reorder atoms positions and radii and fill cellStartEnd
    sortCell <<< N , NBTHREADS>>>(N, cudaAtomPosRad, cudaHashIndex, cudaSortedAtomPosRad, cellStartEnd);

    gpuErrchk( cudaPeekAtLastError() );

    gpuErrchk( cudaFree(cudaAtomPosRad) );


    //-------------- Step 2 : Compute points of the grid outside or inside the surface -----------------
    //Use slices of the grid to avoid allocating large amount of data
    int rangeSearchRefine = (int)ceil(PROBERADIUS / resoSES);
    int sliceSmallSize = min(SLICE , gridSESSize);
    int sliceSize = min(SLICE + 2 * rangeSearchRefine, gridSESSize);
    // int sliceSmallNbCellSES = sliceSmallSize * sliceSmallSize * sliceSmallSize;
    int sliceNbCellSES = sliceSize * sliceSize * sliceSize;
    // int3 sliceGridSESDim = make_int3(sliceSmallSize, sliceSmallSize, sliceSmallSize);
    int3 fullSliceGridSESDim = make_int3(sliceSize, sliceSize, sliceSize);

    gpuErrchk(cudaMalloc((void **)&cudaGridValues, sizeof(float) * sliceNbCellSES));
    gpuErrchk(cudaMalloc((void **)&cudaFillCheck, sizeof(int) * sliceNbCellSES));

    gpuErrchk( cudaMalloc(&vertPerCell, sizeof(uint2) * sliceNbCellSES) );
    gpuErrchk( cudaMalloc(&compactedVoxels, sizeof(unsigned int) * sliceNbCellSES) );

    gpuErrchk( cudaPeekAtLastError() );

    // cerr << "Allocating " << (sizeof(float) * sliceNbCellSES + 3 * sizeof(int) * sliceNbCellSES) / 1000000.0f << " Mo" << endl;

    int3 offset = {0, 0, 0};
    int cut = 8;

    float4 sphereInclusionRad = make_float4(1.9f, 26.7f, -21.91f, 8.0f);
    float3 sphereCenter = make_float3(sphereInclusionRad.x, sphereInclusionRad.y, sphereInclusionRad.z);

    unsigned long long int totalCavityCount = 0;

    cerr << "Full size grid = " << gridSESSize << " x " << gridSESSize << " x " << gridSESSize << endl;
    // cudaEventRecord(start);
    // for (int slice = 0; slice < gridSESSize; slice += sliceSmallSize) {
    for (int i = 0; i < gridSESSize; i += sliceSmallSize) {
        offset.x = i;
        for (int j = 0; j < gridSESSize; j += sliceSmallSize) {
            offset.y = j;
            for (int k = 0; k < gridSESSize; k += sliceSmallSize) {
                offset.z = k;
                // cerr << "Starting : " << offset.x << " / " << offset.y << " / " << offset.z << endl;

                memsetCudaFloat <<< (sliceNbCellSES + NBTHREADS - 1) / NBTHREADS, NBTHREADS >>> (cudaGridValues, OUTSIDE, sliceNbCellSES);
                memsetCudaInt <<< (sliceNbCellSES + NBTHREADS - 1) / NBTHREADS, NBTHREADS >>> (cudaFillCheck, EMPTYCELL, sliceNbCellSES);

                dim3 localWorkSize(cut, cut, cut);
                // dim3 globalWorkSize((sliceSmallSize + cut - 1) / cut, (sliceSmallSize + cut - 1) / cut, (sliceSmallSize + cut - 1) / cut);
                dim3 globalWorkSize((sliceSize + cut - 1) / cut, (sliceSize + cut - 1) / cut, (sliceSize + cut - 1) / cut);


                int3 reducedOffset = make_int3(max(0, offset.x - rangeSearchRefine),
                                               max(0, offset.y - rangeSearchRefine),
                                               max(0, offset.z - rangeSearchRefine));

                cerr << "Fulllll : " << fullSliceGridSESDim.x << ", " << fullSliceGridSESDim.y << ", " << fullSliceGridSESDim.z << endl;
                // cerr << "global = " << globalWorkSize.x << ", " << globalWorkSize.y << ", " << globalWorkSize.z << "   " << (sliceSmallSize + cut - 1) / cut << endl;


                probeIntersection <<< globalWorkSize, localWorkSize >>>(cudaFillCheck, cudaHashIndex, gridNeighborDim, originGridNeighborDx,
                        gridSESDim, fullSliceGridSESDim, originGridSESDx, cellStartEnd,
                        cudaSortedAtomPosRad, cudaGridValues, /*offset*/ reducedOffset, N, sliceNbCellSES);

                gpuErrchk( cudaPeekAtLastError() );
                gpuErrchk( cudaDeviceSynchronize() );



                thrust::device_ptr<int> fillThrust(cudaFillCheck);
                thrust::sort(fillThrust, fillThrust + sliceNbCellSES);

                unsigned int notEmptyCells = thrust::count_if(thrust::device, fillThrust, fillThrust + sliceNbCellSES, is_notempty());


                if (notEmptyCells == 0) {
                    // cerr << "Empty cells !!!" << endl;
                    continue;
                }

                localWorkSize = dim3(NBTHREADS, 1.0f,  1.0f);

                //Too long execution of this kernel triggers the watchdog timer => cut it
                int tranche = min(notEmptyCells, 65536 / 8 * NBTHREADS);

                const int nbStream = 4;
                cudaStream_t streams[nbStream];
                for (int i = 0; i < nbStream; i++)
                    cudaStreamCreate(&(streams[i]));
                int idStream = 0;

                for (unsigned int o = 0; o < notEmptyCells; o += tranche) {

                    globalWorkSize = dim3((tranche + NBTHREADS - 1) / NBTHREADS, 1.0f, 1.0f);
                    // cerr <<o<< " Launch (" << globalWorkSize.x << ", "<<globalWorkSize.y<<", "<<globalWorkSize.z<<") x ("<<localWorkSize.x<<", "<<localWorkSize.y<<", 1.0)" << endl;

                    distanceFieldRefine <<< globalWorkSize, localWorkSize, 0, streams[idStream]>>> (cudaFillCheck, cudaHashIndex, gridNeighborDim, originGridNeighborDx,
                            gridSESDim, fullSliceGridSESDim/*sliceGridSESDim*/, originGridSESDx, cellStartEnd,
                            cudaSortedAtomPosRad, cudaGridValues, N, notEmptyCells, reducedOffset, o);

                    idStream++;
                    if (idStream == nbStream)
                        idStream = 0;

                }

                gpuErrchk( cudaPeekAtLastError() );
                gpuErrchk( cudaDeviceSynchronize() );

                for (int i = 0; i < nbStream; i++)
                    cudaStreamDestroy(streams[i]);



                localWorkSize = dim3(cut, cut, cut);

                globalWorkSize = dim3((sliceSize + cut - 1) / cut, (sliceSize + cut - 1) / cut, (sliceSize + cut - 1) / cut);

                filterNotInSphere <<< globalWorkSize, localWorkSize >>>(sphereInclusionRad, cudaFillCheck, 
                                  originGridNeighborDx, gridSESDim, fullSliceGridSESDim, originGridSESDx,
                                  cudaGridValues, reducedOffset);

                gpuErrchk( cudaPeekAtLastError() );
                gpuErrchk( cudaDeviceSynchronize() );

                thrust::device_ptr<float> valuesThrust(cudaGridValues);
                unsigned int isCavity = thrust::count_if(thrust::device, valuesThrust, valuesThrust + sliceNbCellSES, is_cavity());


                // if(false){
                //     float *hostValues = (float *)malloc(sizeof(float) * sliceNbCellSES);
                //     cudaMemcpy(hostValues, cudaGridValues, sizeof(float) * sliceNbCellSES, cudaMemcpyDeviceToHost);

                //     unsigned int cpt = 0;
                //     int tmp = 0;
                //     for(int a=0;a<sliceNbCellSES;a++){
                //         if(hostValues[a] > 0.0f && hostValues[a] < probeRadius ){
                //             int3 ijk = unflatten1DTo3D(a, fullSliceGridSESDim);
                //             int3 ijkOffset = make_int3(ijk.x + reducedOffset.x, ijk.y + reducedOffset.y, ijk.z + reducedOffset.z);
                //             float3 spacePos3DCellSES = gridToSpace(ijkOffset, originGridNeighbor, resoSES);
                //             printf("HETATM 1208  ND  HEC A  90    %8.3f%8.3f%8.3f\n",spacePos3DCellSES.x ,spacePos3DCellSES.y , spacePos3DCellSES.z );
                //             cpt++;
                //         }

                //     }
                //     // totalCavityCount += cpt;
                // }
                totalCavityCount += isCavity;


// // // Debug output to DX file
//                 float *hostValues2 = (float *)malloc(sizeof(float) * sliceNbCellSES);
//                 cudaMemcpy(hostValues2, cudaGridValues, sizeof(float) * sliceNbCellSES, cudaMemcpyDeviceToHost);

//                         if(i == 0 && j == 0 && k == 0){
//                     FILE *fptr;
//                     string filename = "debug.dx";
//                     if ((fptr = fopen(filename.c_str(), "w")) != NULL) {
//                         fprintf(fptr, "object 1 class gridpositions counts %d %d %d\n", sliceSize, sliceSize, sliceSize);
//                         fprintf(fptr, "origin %f %f %f\n",originGridSESDx.x, originGridSESDx.y, originGridSESDx.z );
//                         fprintf(fptr, "delta %f %f %f\n", resoSES, 0.0f, 0.0f);
//                         fprintf(fptr, "delta %f %f %f\n", 0.0f, resoSES, 0.0f);
//                         fprintf(fptr, "delta %f %f %f\n", 0.0f, 0.0f, resoSES);
//                         fprintf(fptr, "object 2 class gridconnections counts %d %d %d\n", sliceSize, sliceSize, sliceSize);
//                         fprintf(fptr, "object 3 class array type double rank 0 items %d data follows\n", sliceNbCellSES);
//                         for(int a = 0; a < sliceNbCellSES; a++){
//                             if(a != 0 && a%3 == 0){
//                                 fprintf(fptr, "\n");
//                             }
//                             fprintf(fptr, "%f ", hostValues2[a]);
//                         }
//                         fprintf(fptr, "\nattribute \"dep\" string \"positions\"\nobject \"regular positions regular connections\" class field\ncomponent \"positions\" value 1\ncomponent \"connections\" value 2\ncomponent \"data\" value 3\n");

//                         fclose(fptr);
//                     }
//                 }

            }
            // break;
        }
        // break;
    }
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cerr << "Time for step 2 : " << milliseconds << " ms" << std::endl;


    cudaFree(cudaSortedAtomPosRad);
    cudaFree(cudaHashIndex);
    cudaFree(cellStartEnd);
    cudaFree(cudaGridValues);
    cudaFree(cudaFillCheck);

    free(atomPosRad);

    double cavVolume = totalCavityCount * (resoSES * resoSES * resoSES);

    cerr << "CAVITY VOLUME = "<<cavVolume<< " A^3"<<endl;

}




int main(int argc, const char * argv[]) {

    args::ArgumentParser parser("QuickSES, SES mesh generation using GPU", "");
    args::Group groupMandatory(parser, "", args::Group::Validators::All);
    args::Group groupOptional(parser,  "", args::Group::Validators::DontCare);
    args::ValueFlag<string> inFile(groupMandatory, "input.pdb", "Input PDB file", {'i'});
    // args::ValueFlag<string> outFile(groupMandatory, "output.obj", "Output OBJ mesh file", {'o'});
    // args::ValueFlag<int> smoothTimes(groupOptional, "smooth factor", "(1) Times to run Laplacian smoothing step.", {'l'});
    args::ValueFlag<float> voxelSize(groupOptional, "voxel size", "(0.5) Voxel size in Angstrom. Defines the quality of the mesh.", {'v'});
    args::ValueFlag<int> slice(groupOptional, "slice size", "(300) Size of the sub-grid. Defines the quantity of GPU memory needed.", {'s'});
    args::HelpFlag help(groupOptional, "help", "   Display this help menu", {'h', "help"});

    try {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help) {
        std::cerr << parser;
        return 0;
    }
    catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return -1;
    }
    catch (args::ValidationError e) {
        // std::cerr << e.what() << std::endl;
        std::cerr << "Usage: " << parser;
        return -1;
    }

    if (inFile) { inputFilePath = args::get(inFile); }
    // if (outFile) { outputFilePath = args::get(outFile); }
    // if (smoothTimes) { laplacianSmoothSteps = args::get(smoothTimes); }
    if (voxelSize) { gridResolutionSES = args::get(voxelSize); }
    if (slice) {SLICE = args::get(slice); }

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

    for (int chainId = 0; chainId < P->size; chainId++) {
        C = &P->chains[chainId];

        A = &C->residues[0].atoms[0];

        while (A != NULL) {
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


    // std::vector<MeshData> resultMeshes = computeSlicedSES(&atomPos[0], &atomRadii[0], N, gridResolutionSES, laplacianSmoothSteps);
    // std::vector<MeshData> resultMeshes = computeSlicedSESCPU(P);

    computePocketVolume(atomPos.data(), atomRadii.data(), N, gridResolutionSES);

    //Write to OBJ
    // writeToObj(outputFilePath, resultMeshes);

    freePDB(P);

    return 0;
}
