// countVertexPerCell  (MarchingCubes.cu:571)
// offset = 0, sliceGrid == sesGrid. Writes vertPerCell[id] = {numVerts, occupied}.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> gridValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> vertPerCell: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> nbTriTable: array<i32>; // 256

fn to1D(ids: vec3<i32>, dim: vec3<i32>) -> i32 {
    return (dim.y * dim.z * ids.x) + (dim.z * ids.y) + ids.z;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    let k = i32(gid.z);
    let dim = P.sliceGridDim;
    let rs = P.rangeSearchRefine;
    let off = P.offset;

    if (i > dim.x - 2) { return; }
    if (j > dim.y - 2) { return; }
    if (k > dim.z - 2) { return; }

    // Seam dedup: low-edge halo skip when this slab is NOT first along an axis.
    if (off.x != 0 && i < rs - 3) { return; }
    if (off.y != 0 && j < rs - 3) { return; }
    if (off.z != 0 && k < rs - 3) { return; }

    // When this slab IS first along an axis (offset==0), trim its high edge.
    if (off.x == 0 && i >= dim.x - rs * 2 - 1) { return; }
    if (off.y == 0 && j >= dim.y - rs * 2 - 1) { return; }
    if (off.z == 0 && k >= dim.z - rs * 2 - 1) { return; }

    // Always skip the high-edge halo.
    if (i >= dim.x - rs - 1) { return; }
    if (j >= dim.y - rs - 1) { return; }
    if (k >= dim.z - rs - 1) { return; }

    let id = to1D(vec3<i32>(i, j, k), dim);

    let v0 = gridValues[to1D(vec3<i32>(i,     j,     k    ), dim)];
    let v1 = gridValues[to1D(vec3<i32>(i + 1, j,     k    ), dim)];
    let v2 = gridValues[to1D(vec3<i32>(i + 1, j + 1, k    ), dim)];
    let v3 = gridValues[to1D(vec3<i32>(i,     j + 1, k    ), dim)];
    let v4 = gridValues[to1D(vec3<i32>(i,     j,     k + 1), dim)];
    let v5 = gridValues[to1D(vec3<i32>(i + 1, j,     k + 1), dim)];
    let v6 = gridValues[to1D(vec3<i32>(i + 1, j + 1, k + 1), dim)];
    let v7 = gridValues[to1D(vec3<i32>(i,     j + 1, k + 1), dim)];

    let iso = 0.0;
    var cubeIndex: i32 = 0;
    cubeIndex = cubeIndex + i32(v0 < iso);
    cubeIndex = cubeIndex + i32(v1 < iso) * 2;
    cubeIndex = cubeIndex + i32(v2 < iso) * 4;
    cubeIndex = cubeIndex + i32(v3 < iso) * 8;
    cubeIndex = cubeIndex + i32(v4 < iso) * 16;
    cubeIndex = cubeIndex + i32(v5 < iso) * 32;
    cubeIndex = cubeIndex + i32(v6 < iso) * 64;
    cubeIndex = cubeIndex + i32(v7 < iso) * 128;

    let nv = u32(nbTriTable[cubeIndex]);
    var occ: u32 = 0u;
    if (nv > 0u) { occ = 1u; }
    vertPerCell[id] = vec2<u32>(nv, occ);
}
