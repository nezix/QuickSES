
#define MAX_VERTEX 15
// __constant__ int MAX_VERTEX = 15;


__constant__ unsigned int edgeTable[256] =
{
	0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
	0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
	0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
	0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
	0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
	0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
	0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
	0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
	0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
	0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
	0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
	0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
	0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
	0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
	0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
	0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
	0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
	0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

__constant__ int triTable[256][16] = {
	{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
	{ 8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1 },
	{ 3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1 },
	{ 4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
	{ 4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1 },
	{ 9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1 },
	{ 10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1 },
	{ 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
	{ 5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1 },
	{ 8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1 },
	{ 2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
	{ 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1 },
	{ 11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1 },
	{ 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1 },
	{ 11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1 },
	{ 11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1 },
	{ 2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1 },
	{ 6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
	{ 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1 },
	{ 6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
	{ 6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1 },
	{ 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1 },
	{ 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1 },
	{ 3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
	{ 0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1 },
	{ 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1 },
	{ 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
	{ 5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1 },
	{ 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1 },
	{ 6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1 },
	{ 10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
	{ 1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1 },
	{ 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1 },
	{ 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
	{ 6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1 },
	{ 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1 },
	{ 8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1 },
	{ 3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1 },
	{ 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1 },
	{ 10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
	{ 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1 },
	{ 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
	{ 2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1 },
	{ 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1 },
	{ 11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1 },
	{ 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1 },
	{ 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1 },
	{ 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1 },
	{ 7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1 },
	{ 10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1 },
	{ 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1 },
	{ 7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1 },
	{ 6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1 },
	{ 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1 },
	{ 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1 },
	{ 8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1 },
	{ 1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1 },
	{ 10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1 },
	{ 10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1 },
	{ 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1 },
	{ 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1 },
	{ 3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1 },
	{ 7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1 },
	{ 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1 },
	{ 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1 },
	{ 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1 },
	{ 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1 },
	{ 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1 },
	{ 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1 },
	{ 6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1 },
	{ 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1 },
	{ 6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1 },
	{ 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1 },
	{ 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1 },
	{ 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1 },
	{ 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1 },
	{ 9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1 },
	{ 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1 },
	{ 10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1 },
	{ 0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1 },
	{ 11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1 },
	{ 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1 },
	{ 7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1 },
	{ 2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1 },
	{ 9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1 },
	{ 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1 },
	{ 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1 },
	{ 0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1 },
	{ 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1 },
	{ 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1 },
	{ 0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1 },
	{ 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1 },
	{ 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1 },
	{ 5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1 },
	{ 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1 },
	{ 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1 },
	{ 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1 },
	{ 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1 },
	{ 3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1 },
	{ 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1 },
	{ 9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1 },
	{ 11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1 },
	{ 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1 },
	{ 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1 },
	{ 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1 },
	{ 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1 },
	{ 4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1 },
	{ 0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1 },
	{ 1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }
};

__constant__ int nbTriTable[256] {
	0, 3, 3, 6, 3, 6, 6, 9, 3, 6, 6, 9, 6, 9, 9, 6, 3,
	6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
	3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12,
	9, 6, 9, 9, 6, 9, 12, 12, 9, 9, 12, 12, 9, 12,
	15, 15, 6, 3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12,
	9, 12, 12, 9, 6, 9, 9, 12, 9, 12, 12, 15, 9, 12,
	12, 15, 12, 15, 15, 12, 6, 9, 9, 12, 9, 12, 6,
	9, 9, 12, 12, 15, 12, 15, 9, 6, 9, 12, 12, 9, 12,
	15, 9, 6, 12, 15, 15, 12, 15, 6, 12, 3, 3, 6, 6,
	9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9, 6, 9,
	9, 12, 9, 12, 12, 15, 9, 6, 12, 9, 12, 9, 15, 6,
	6, 9, 9, 12, 9, 12, 12, 15, 9, 12, 12, 15, 12,
	15, 15, 12, 9, 12, 12, 9, 12, 15, 15, 12, 12,
	9, 15, 6, 15, 12, 6, 3, 6, 9, 9, 12, 9, 12, 12,
	15, 9, 12, 12, 15, 6, 9, 9, 6, 9, 12, 12, 15,
	12, 15, 15, 6, 12, 9, 15, 12, 9, 6, 12, 3, 9,
	12, 12, 15, 12, 15, 9, 12, 12, 15, 15, 6, 9,
	12, 6, 3, 6, 9, 9, 6, 9, 12, 6, 3, 9, 6, 12, 3,
	6, 3, 3, 0
};

struct point {
	int3 xyz;
	float val;
};


__device__ float3 linearInterpolation(float isovalue, point voxelA, point voxelB) {

	float mu;
	float3 p;

	if (abs(isovalue - voxelA.val) < EPSILON) {
		p.x = (float)voxelA.xyz.x;
		p.y = (float)voxelA.xyz.y;
		p.z = (float)voxelA.xyz.z;
		return (p);
	}
	if (abs(isovalue - voxelB.val) < EPSILON) {
		p.x = (float)voxelB.xyz.x;
		p.y = (float)voxelB.xyz.y;
		p.z = (float)voxelB.xyz.z;
		return (p);
	}
	if (abs(voxelA.val - voxelB.val) < EPSILON) {
		p.x = (float)voxelA.xyz.x;
		p.y = (float)voxelA.xyz.y;
		p.z = (float)voxelA.xyz.z;
		return (p);
	}

	mu = (isovalue - voxelA.val) / (voxelB.val - voxelA.val);

	p.x = voxelA.xyz.x + mu * (voxelB.xyz.x - voxelA.xyz.x);
	p.y = voxelA.xyz.y + mu * (voxelB.xyz.y - voxelA.xyz.y);
	p.z = voxelA.xyz.z + mu * (voxelB.xyz.z - voxelA.xyz.z);

	return p;


	// const float scale = (isovalue - voxelA.val) / (voxelB.val - voxelA.val);

	// //Coordinates of position will be in float3
	// float3 position;

	// //Initialising position
	// position.x = voxelA.xyz.x + scale * (voxelB.xyz.x - voxelA.xyz.x);
	// position.y = voxelA.xyz.y + scale * (voxelB.xyz.y - voxelA.xyz.y);
	// position.z = voxelA.xyz.z + scale * (voxelB.xyz.z - voxelA.xyz.z);

	// return position;
}

__device__ inline unsigned int to1D(int3 ids, int3 dim) {
	return (dim.y * dim.z * ids.x) + (dim.z * ids.y) + ids.z;
}
__device__ inline unsigned int to1D(uint3 ids, int3 dim) {
	return (dim.y * dim.z * ids.x) + (dim.z * ids.y) + ids.z;
}

__global__ void MCkernel(float isovalue, int3 dim, float* data, float3* vertex, int* triangle, int3 offset, unsigned int nbcells)
{


	unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x) + offset.x;
	unsigned int j = (blockIdx.y * blockDim.y + threadIdx.y) + offset.y;
	unsigned int k = (blockIdx.z * blockDim.z + threadIdx.z) + offset.z;

	if (i >= dim.x - 1)
		return;
	if (j >= dim.y - 1)
		return;
	if (k >= dim.z - 1)
		return;


	//Variables
	point voxels[8];
	float3 vertlist[12];
	int index[8];
	int cubeIndex = 0;
	float3 vertices[MAX_VERTEX];
	int numTriangles = 0;
	int numVertices = 0;


	index[0] = to1D(make_int3(i, j, k), dim);
	index[1] = to1D(make_int3(i + 1, j, k), dim);
	index[2] = to1D(make_int3(i + 1, j + 1, k), dim);
	index[3] = to1D(make_int3(i, j + 1, k), dim);
	index[4] = to1D(make_int3(i, j, k + 1), dim);
	index[5] = to1D(make_int3(i + 1, j, k + 1), dim);
	index[6] = to1D(make_int3(i + 1, j + 1, k + 1), dim);
	index[7] = to1D(make_int3(i, j + 1, k + 1), dim);

	voxels[0].xyz.x = i;
	voxels[0].xyz.y = j;
	voxels[0].xyz.z = k;
	voxels[0].val = data[index[0]];

	voxels[1].xyz.x = i + 1;
	voxels[1].xyz.y = j;
	voxels[1].xyz.z = k;
	voxels[1].val = data[index[1]];


	voxels[2].xyz.x = i + 1;
	voxels[2].xyz.y = j + 1;
	voxels[2].xyz.z = k;
	voxels[2].val = data[index[2]];

	voxels[3].xyz.x = i;
	voxels[3].xyz.y = j + 1;
	voxels[3].xyz.z = k;
	voxels[3].val = data[index[3]];

	voxels[4].xyz.x = i;
	voxels[4].xyz.y = j;
	voxels[4].xyz.z = k + 1;
	voxels[4].val = data[index[4]];

	voxels[5].xyz.x = i + 1;
	voxels[5].xyz.y = j;
	voxels[5].xyz.z = k + 1;
	voxels[5].val = data[index[5]];

	voxels[6].xyz.x = i + 1;
	voxels[6].xyz.y = j + 1;
	voxels[6].xyz.z = k + 1;
	voxels[6].val = data[index[6]];

	voxels[7].xyz.x = i;
	voxels[7].xyz.y = j + 1;
	voxels[7].xyz.z = k + 1;
	voxels[7].val = data[index[7]];


	//PolygoniseCube
	if (voxels[0].val < isovalue) cubeIndex |= 1;
	if (voxels[1].val < isovalue) cubeIndex |= 2;
	if (voxels[2].val < isovalue) cubeIndex |= 4;
	if (voxels[3].val < isovalue) cubeIndex |= 8;
	if (voxels[4].val < isovalue) cubeIndex |= 16;
	if (voxels[5].val < isovalue) cubeIndex |= 32;
	if (voxels[6].val < isovalue) cubeIndex |= 64;
	if (voxels[7].val < isovalue) cubeIndex |= 128;


	//Getting edges
	unsigned int edges = edgeTable[cubeIndex];

	//Comparing edges with 12 bit by and operation and position coordinate
	if (edges == 0) {
		return;
	}
	if (edges & 1) {
		vertlist[0] = linearInterpolation(isovalue, voxels[0], voxels[1]);
	}
	if (edges & 2) {
		vertlist[1] = linearInterpolation(isovalue, voxels[1], voxels[2]);
	}
	if (edges & 4) {
		vertlist[2] = linearInterpolation(isovalue, voxels[2], voxels[3]);
	}
	if (edges & 8) {
		vertlist[3] = linearInterpolation(isovalue, voxels[3], voxels[0]);
	}
	if (edges & 16) {
		vertlist[4] = linearInterpolation(isovalue, voxels[4], voxels[5]);
	}
	if (edges & 32) {
		vertlist[5] = linearInterpolation(isovalue, voxels[5], voxels[6]);
	}
	if (edges & 64) {
		vertlist[6] = linearInterpolation(isovalue, voxels[6], voxels[7]);
	}
	if (edges & 128) {
		vertlist[7] = linearInterpolation(isovalue, voxels[7], voxels[4]);
	}
	if (edges & 256) {
		vertlist[8] = linearInterpolation(isovalue, voxels[0], voxels[4]);
	}
	if (edges & 512) {
		vertlist[9] = linearInterpolation(isovalue, voxels[1], voxels[5]);
	}
	if (edges & 1024) {
		vertlist[10] = linearInterpolation(isovalue, voxels[2], voxels[6]);
	}
	if (edges & 2048) {
		vertlist[11] = linearInterpolation(isovalue, voxels[3], voxels[7]);
	}

	if (cubeIndex >= 0 && cubeIndex < 256 ) {
		for (int i = 0; i < nbTriTable[cubeIndex]; i += 3) {
			if (triTable[cubeIndex][i] == -1 || numVertices + 3 >= MAX_VERTEX)
				break;
			vertices[numVertices++] = vertlist[triTable[cubeIndex][i]];
			vertices[numVertices++] = vertlist[triTable[cubeIndex][i + 1]];
			vertices[numVertices++] = vertlist[triTable[cubeIndex][i + 2]];
			++numTriangles;
		}
	}

	// for (int n = 0; n < MAX_VERTEX; n += 3)
	// {
	// 	int edgeNumber = triTable[cubeIndex][n];
	// 	if (edgeNumber < 0)
	// 		break;

	// 	vertices[numVertices++] = pos[edgeNumber];
	// 	vertices[numVertices++] = pos[triTable[cubeIndex][n + 1]];
	// 	vertices[numVertices++] = pos[triTable[cubeIndex][n + 2]];
	// 	++numTriangles;
	// }

	//Getting the number of triangles
	triangle[index[0]] = numTriangles;

	//Vertex List
	for (int n = 0; n < min(numVertices, MAX_VERTEX); ++n) {
		vertex[MAX_VERTEX * index[0] + n] = vertices[n];
	}
}




__global__ void countVertexPerCell(const float isovalue, const int3 dim, const float* data, uint2 *vertPerCell) {


	unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned int j = (blockIdx.y * blockDim.y + threadIdx.y);
	unsigned int k = (blockIdx.z * blockDim.z + threadIdx.z);

	if (i >= dim.x)
		return;
	if (j >= dim.y)
		return;
	if (k >= dim.z)
		return;


	unsigned int id = to1D(make_int3(i, j, k), dim);
	uint2 Nverts = make_uint2(0, 0);

	if (i > (dim.x - 2) || j > (dim.y - 2) || k > (dim.z - 2)) {
		vertPerCell[id] = Nverts;
		return;
	}

	float voxel0;
	float voxel1;
	float voxel2;
	float voxel3;
	float voxel4;
	float voxel5;
	float voxel6;
	float voxel7;

	// __shared__ float voxels[8];
	// float3 vertlist[12];
	// int index[8];
	int cubeIndex = 0;

	voxel0 = data[to1D(make_int3(i, j, k), dim)];
	voxel1 = data[to1D(make_int3(i + 1, j, k), dim)];
	voxel2 = data[to1D(make_int3(i + 1, j + 1, k), dim)];
	voxel3 = data[to1D(make_int3(i, j + 1, k), dim)];
	voxel4 = data[to1D(make_int3(i, j, k + 1), dim)];
	voxel5 = data[to1D(make_int3(i + 1, j, k + 1), dim)];
	voxel6 = data[to1D(make_int3(i + 1, j + 1, k + 1), dim)];
	voxel7 = data[to1D(make_int3(i, j + 1, k + 1), dim)];


	//PolygoniseCube
	// if (voxel0 < isovalue) cubeIndex |= 1;
	// if (voxel1 < isovalue) cubeIndex |= 2;
	// if (voxel2 < isovalue) cubeIndex |= 4;
	// if (voxel3 < isovalue) cubeIndex |= 8;
	// if (voxel4 < isovalue) cubeIndex |= 16;
	// if (voxel5 < isovalue) cubeIndex |= 32;
	// if (voxel6 < isovalue) cubeIndex |= 64;
	// if (voxel7 < isovalue) cubeIndex |= 128;

	cubeIndex =  ((unsigned int) (voxel0 < isovalue));
	cubeIndex += ((unsigned int) (voxel1 < isovalue))* 2;
	cubeIndex += ((unsigned int) (voxel2 < isovalue))* 4;
	cubeIndex += ((unsigned int) (voxel3 < isovalue))* 8;
	cubeIndex += ((unsigned int) (voxel4 < isovalue))* 16;
	cubeIndex += ((unsigned int) (voxel5 < isovalue))* 32;
	cubeIndex += ((unsigned int) (voxel6 < isovalue))* 64;
	cubeIndex += ((unsigned int) (voxel7 < isovalue))* 128;

	Nverts.x = nbTriTable[cubeIndex];
	Nverts.y = (Nverts.x > 0);
	vertPerCell[id] = Nverts;

}

__global__ void compactVoxels(unsigned int * compactedVoxelArray,
                              const uint2 *voxelOccupied,
                              unsigned int lastVoxel, unsigned int numVoxels,
                              unsigned int numVoxelsp1, int3 dim)
{


	unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned int j = (blockIdx.y * blockDim.y + threadIdx.y);
	unsigned int k = (blockIdx.z * blockDim.z + threadIdx.z);

	if (i >= dim.x)
		return;
	if (j >= dim.y)
		return;
	if (k >= dim.z)
		return;


	unsigned int id = to1D(make_int3(i, j, k), dim);

	if (id < numVoxels) {
		if ( (id < numVoxelsp1) ? voxelOccupied[id].y < voxelOccupied[id + 1].y : lastVoxel ) {
			compactedVoxelArray[ voxelOccupied[id].y ] = id;
		}
	}
}
inline __device__ float3 lerp(float3 a, float3 b, float t)
{
	return a + t * (b - a);
}

__device__ float3 gridPosition(uint3 cellPos, float3 originGrid, float dx) {
	return (originGrid + (make_float3(cellPos.x, cellPos.y, cellPos.z) * dx) );
}
__device__ uint3 grid1DTo3D(unsigned int index, int3 gridDim) {
	uint3 res;
	res.x = index / (gridDim.y * gridDim.z); //Note the integer division . This is x
	res.y = (index - res.x * gridDim.y * gridDim.z) / gridDim.z; //This is y
	res.z = index - res.x * gridDim.y * gridDim.z - res.y * gridDim.z; //This is z
	return res;
}

__device__ float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1) {
	float t = (isolevel - f0) / (f1 - f0);
	return lerp(p0, p1, t);
}

__global__ void generateTriangleVerticesSMEM(float3 *pos,
        const unsigned int *compactedVoxelArray,
        const uint2 * numVertsScanned,
        const float *gridValues,
        float4 originGridDX,
        float isoValue, unsigned int activeVoxels,
        unsigned int maxVertsM3, int3 dim, int3 offset)
{


	unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned int j = (blockIdx.y * blockDim.y + threadIdx.y);
	unsigned int k = (blockIdx.z * blockDim.z + threadIdx.z);

	// if (i >= dim.x)
	// 	return;
	// if (j >= dim.y)
	// 	return;
	// if (k >= dim.z)
	// 	return;
	int3 cudaGrid = make_int3(blockDim.x, blockDim.y, blockDim.z);
	unsigned int id = to1D(make_int3(i, j, k), cudaGrid);

	if (id >= activeVoxels)
		return;

	unsigned int voxel = compactedVoxelArray[id];
	uint3 gridPos = grid1DTo3D(voxel, dim);

	float dx = originGridDX.w;
	float3 originGrid = make_float3(originGridDX.x, originGridDX.y, originGridDX.z);

	float3 p = gridPosition(gridPos, originGrid, dx);


	// calculate cell vertex positions
	float3 v[8];
	v[0] = p;
	v[1] = p + make_float3(dx, 0, 0);
	v[2] = p + make_float3(dx, dx, 0);
	v[3] = p + make_float3(0, dx, 0);
	v[4] = p + make_float3(0, 0, dx);
	v[5] = p + make_float3(dx, 0, dx);
	v[6] = p + make_float3(dx, dx, dx);
	v[7] = p + make_float3(0, dx, dx);

	float field[8];
	field[0] = gridValues[voxel];
	field[1] = gridValues[to1D(gridPos + make_uint3(1, 0, 0), dim)];
	field[2] = gridValues[to1D(gridPos + make_uint3(1, 1, 0), dim)];
	field[3] = gridValues[to1D(gridPos + make_uint3(0, 1, 0), dim)];
	field[4] = gridValues[to1D(gridPos + make_uint3(0, 0, 1), dim)];
	field[5] = gridValues[to1D(gridPos + make_uint3(1, 0, 1), dim)];
	field[6] = gridValues[to1D(gridPos + make_uint3(1, 1, 1), dim)];
	field[7] = gridValues[to1D(gridPos + make_uint3(0, 1, 1), dim)];

	// recalculate flag
	unsigned int cubeindex;
	cubeindex =  ((unsigned int)(field[0] < isoValue));
	cubeindex += ((unsigned int)(field[1] < isoValue)) * 2;
	cubeindex += ((unsigned int)(field[2] < isoValue)) * 4;
	cubeindex += ((unsigned int)(field[3] < isoValue)) * 8;
	cubeindex += ((unsigned int)(field[4] < isoValue)) * 16;
	cubeindex += ((unsigned int)(field[5] < isoValue)) * 32;
	cubeindex += ((unsigned int)(field[6] < isoValue)) * 64;
	cubeindex += ((unsigned int)(field[7] < isoValue)) * 128;

	// find the vertices where the surface intersects the cube
	// Note: SIMD marching cubes implementations have no need
	//       for an edge table, because branch divergence eliminates any
	//       potential performance gain from only computing the per-edge
	//       vertices when indicated by the edgeTable.

	// Use shared memory to keep register pressure under control.
	// No need to call __syncthreads() since each thread uses its own
	// private shared memory buffer.
	__shared__ float3 vertlist[12 * NBTHREADS];

	vertlist[threadIdx.x                  ]  = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
	vertlist[(NBTHREADS * 1) + threadIdx.x]  = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
	vertlist[(NBTHREADS * 2) + threadIdx.x]  = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
	vertlist[(NBTHREADS * 3) + threadIdx.x]  = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);
	vertlist[(NBTHREADS * 4) + threadIdx.x]  = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
	vertlist[(NBTHREADS * 5) + threadIdx.x]  = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
	vertlist[(NBTHREADS * 6) + threadIdx.x]  = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
	vertlist[(NBTHREADS * 7) + threadIdx.x]  = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);
	vertlist[(NBTHREADS * 8) + threadIdx.x]  = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
	vertlist[(NBTHREADS * 9) + threadIdx.x]  = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
	vertlist[(NBTHREADS * 10) + threadIdx.x] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
	vertlist[(NBTHREADS * 11) + threadIdx.x] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);

	float3 offsetPos = make_float3(offset.x * dx, offset.y * dx, offset.z * dx);

	// output triangle vertices
	unsigned int numVerts = nbTriTable[cubeindex];
	for (int i = 0; i < numVerts; i += 3) {
		unsigned int index = numVertsScanned[voxel].x + i;

		float3 *vert[3];
		int edge;
		edge = triTable[cubeindex][i];
		vert[0] = &vertlist[(edge * NBTHREADS) + threadIdx.x];

		edge = triTable[cubeindex][i + 1];
		vert[1] = &vertlist[(edge * NBTHREADS) + threadIdx.x];

		edge = triTable[cubeindex][i + 2];
		vert[2] = &vertlist[(edge * NBTHREADS) + threadIdx.x];

		if (index < maxVertsM3) {
			pos[index  ] = *vert[0] + offsetPos;
			pos[index + 1] = *vert[1] + offsetPos;
			pos[index + 2] = *vert[2] + offsetPos;
		}
	}
}

__global__ void groupVertices(float3 *verts, const unsigned int nbVerts, const float tolerance) {
	unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);

	if (i >= nbVerts)
		return;

	verts[i].x = round(verts[i].x / tolerance) * tolerance;
	verts[i].y = round(verts[i].y / tolerance) * tolerance;
	verts[i].z = round(verts[i].z / tolerance) * tolerance;
}

#define MAXNEIGHBOR 32
__global__ void LaplacianSmooth(float3 *verts, int *triangles, const unsigned int nbVerts, const unsigned int nbTris, const unsigned int ite) {
	//For each vertex
	unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x);

	if (idx >= nbVerts) {
		return;
	}

	/*	int neighbors[MAXNEIGHBOR];
		int curId = -1;

		for (int i = 0; i < MAXNEIGHBOR; i++) {
			neighbors[i] = -1;
		}

		for (int t = 0; t < nbTris && curId < MAXNEIGHBOR; t++) {
			if (triangles[t * 3] == idx || triangles[t * 3 + 1] == idx || triangles[t * 3 + 2] == idx)
				neighbors[curId++] = t;
			// int compar = (triangles[t * 3] == idx || triangles[t * 3 + 1] == idx || triangles[t * 3 + 2] == idx);
			// neighbors[curId*compar] = t;
			// curId = compar + curId;
		}


		for (int t = 0; t < ite; ++t) {


			float3 curV = make_float3(0,0,0);


			for (int i = 0; i < curId; i++) {
				int idv1 = triangles[neighbors[i] * 3 + 0];
				int idv2 = triangles[neighbors[i] * 3 + 1];
				int idv3 = triangles[neighbors[i] * 3 + 2];

				if(idv1 != idx)
					curV = curV + verts[idv1];
				if(idv2 != idx)
					curV = curV + verts[idv2];
				if(idv3 != idx)
					curV = curV + verts[idv3];
			}
			curV = curV / max(1,curId*2);
			verts[idx] = curV;
		}*/

	int neighbors[MAXNEIGHBOR];//The 0 is not used
	int curId = 1;

	for (int i = 0; i < MAXNEIGHBOR; i++) {
		neighbors[i] = -1;
	}

	//TODO: Could use shared mem here

	//Get 31 neighbors max
	for (int t = 0; t < nbTris && curId < MAXNEIGHBOR; t++) {
		int compar = (triangles[t * 3] == idx || triangles[t * 3 + 1] == idx || triangles[t * 3 + 2] == idx);
		neighbors[curId * compar] = t;
		curId = compar + curId;
	}

	for (int t = 0; t < ite; ++t) {
		float3 curV = make_float3(0, 0, 0);
		// float3 save = verts[idx];

		//For all neighbors of current vertex
		for (int i = 1; i < curId ; i++) {
			int idv1 = triangles[neighbors[i] * 3 + 0];
			int idv2 = triangles[neighbors[i] * 3 + 1];
			int idv3 = triangles[neighbors[i] * 3 + 2];

			curV = curV + (verts[idv1] * (idv1 != idx));
			curV = curV + (verts[idv2] * (idv2 != idx));
			curV = curV + (verts[idv3] * (idv3 != idx));
			// curV = curV - save;
		}

		if (curId != 1) {
			curV = (curV / ((curId - 1) * 2));
			verts[idx] = curV;
		}
	}

}

