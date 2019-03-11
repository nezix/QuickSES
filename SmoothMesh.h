#include <vector>
#include "cuda_runtime.h"

struct MeshData
{
    int NVertices = 0;
    int NTriangles = 0;
    float3 *vertices;
    int3 *triangles;
    // std::vector<float3> normals;
    // std::vector<float4> colors;
};


std::vector<std::vector<int>> getNeighboorsVertices(float3 *&vertices, int3 *&triangles, unsigned int vertNum, unsigned int triNum);
std::vector<std::vector<int>> getNeighboorsVertices(const MeshData &mesh);

void smoothMeshLaplacian(int times, float3 *&vertices, int3 *&triangles, unsigned int vertNum, unsigned int triNum);
void smoothMeshLaplacian(int times, MeshData &mesh);

