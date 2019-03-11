#include "SmoothMesh.h"

std::vector<std::vector<int>> getNeighboorsVertices(float3 *&vertices, int3 *&triangles, unsigned int vertNum, unsigned int triNum) {

	std::vector<std::vector<int>> neighboors(vertNum);


	for (unsigned int i = 0; i < triNum; i++) {
		neighboors[triangles[i].x].push_back(triangles[i].y);
		neighboors[triangles[i].x].push_back(triangles[i].z);

		neighboors[triangles[i].y].push_back(triangles[i].x);
		neighboors[triangles[i].y].push_back(triangles[i].z);

		neighboors[triangles[i].z].push_back(triangles[i].x);
		neighboors[triangles[i].z].push_back(triangles[i].y);
	}

	return neighboors;
}

std::vector<std::vector<int>> getNeighboorsVertices(const MeshData &mesh) {

	std::vector<std::vector<int>> neighboors(mesh.NVertices);


	for (unsigned int i = 0; i < mesh.NTriangles; i++) {
		neighboors[mesh.triangles[i].x].push_back(mesh.triangles[i].y);
		neighboors[mesh.triangles[i].x].push_back(mesh.triangles[i].z);

		neighboors[mesh.triangles[i].y].push_back(mesh.triangles[i].x);
		neighboors[mesh.triangles[i].y].push_back(mesh.triangles[i].z);

		neighboors[mesh.triangles[i].z].push_back(mesh.triangles[i].x);
		neighboors[mesh.triangles[i].z].push_back(mesh.triangles[i].y);
	}

	return neighboors;

}

void smoothMeshLaplacian(int times, float3 *&vertices, int3 *&triangles, unsigned int vertNum, unsigned int triNum) {

	std::vector<std::vector<int>> neighboors = getNeighboorsVertices(vertices, triangles, vertNum, triNum);

	float3 newVert = make_float3(0.0f, 0.0f, 0.0f);

	for (int t = 0; t < times; t++) {
		for (unsigned int i = 0; i < vertNum; i++) {
			newVert.x = 0.0f;
			newVert.y = 0.0f;
			newVert.z = 0.0f;

			int nbNeigh = neighboors[i].size();

			for (int n = 0; n < nbNeigh; n++) { //for all neighboors
				// newVert = newVert + vertices[neighboors[i][n]];
				newVert.x = newVert.x + vertices[neighboors[i][n]].x;
				newVert.y = newVert.y + vertices[neighboors[i][n]].y;
				newVert.z = newVert.z + vertices[neighboors[i][n]].z;

			}
			vertices[i].x = newVert.x / (float)nbNeigh;
			vertices[i].y = newVert.y / (float)nbNeigh;
			vertices[i].z = newVert.z / (float)nbNeigh;

		}
	}

}


void smoothMeshLaplacian(int times, MeshData &mesh){

	if(times <= 0){
		return;
	}
	std::vector<std::vector<int>> neighboors = getNeighboorsVertices(mesh);
	float3 newVert = make_float3(0.0f, 0.0f, 0.0f);
	for (int t = 0; t < times; t++) {

		for (unsigned int i = 0; i < mesh.NVertices; i++) {
			newVert.x = 0.0f;
			newVert.y = 0.0f;
			newVert.z = 0.0f;
			
			int nbNeigh = neighboors[i].size();
			for (int n = 0; n < nbNeigh; n++) { //for all neighboors
				// newVert = newVert + vertices[neighboors[i][n]];
				newVert.x = newVert.x + mesh.vertices[neighboors[i][n]].x;
				newVert.y = newVert.y + mesh.vertices[neighboors[i][n]].y;
				newVert.z = newVert.z + mesh.vertices[neighboors[i][n]].z;

			}
			mesh.vertices[i].x = newVert.x / (float)nbNeigh;
			mesh.vertices[i].y = newVert.y / (float)nbNeigh;
			mesh.vertices[i].z = newVert.z / (float)nbNeigh;

		}
	}
}