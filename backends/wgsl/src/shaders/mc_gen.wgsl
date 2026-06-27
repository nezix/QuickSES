// generateTriangleVertices  (MarchingCubes.cu:736), restructured:
// edge vertlist computed in per-thread local array (no workgroup shared mem -> no 36KB limit).
// offset = 0. One thread per active voxel.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> gridValues: array<f32>;
@group(0) @binding(2) var<storage, read> compactedVoxels: array<u32>;
@group(0) @binding(3) var<storage, read> vertOffset: array<u32>;  // exclusive-scan of vertPerCell.x
@group(0) @binding(4) var<storage, read> triTable: array<i32>;    // 256*16
@group(0) @binding(5) var<storage, read> nbTriTable: array<i32>;  // 256
@group(0) @binding(6) var<storage, read_write> outVerts: array<vec4<f32>>; // xyz used

struct GenParams {
    activeVoxels: u32,
    maxVertsM3: u32, // totalVerts - 3
    _p0: u32,
    _p1: u32,
};
@group(0) @binding(7) var<uniform> GP: GenParams;

fn to1Du(ids: vec3<u32>, dim: vec3<i32>) -> i32 {
    let d = vec3<u32>(u32(dim.x), u32(dim.y), u32(dim.z));
    return i32((d.y * d.z * ids.x) + (d.z * ids.y) + ids.z);
}

fn grid1DTo3D(index: u32, dim: vec3<i32>) -> vec3<u32> {
    let d = vec3<u32>(u32(dim.x), u32(dim.y), u32(dim.z));
    let x = index / (d.y * d.z);
    let y = (index - x * d.y * d.z) / d.z;
    let z = index - x * d.y * d.z - y * d.z;
    return vec3<u32>(x, y, z);
}

fn vertexInterp(iso: f32, p0: vec3<f32>, p1: vec3<f32>, f0: f32, f1: f32) -> vec3<f32> {
    let t = (iso - f0) / (f1 - f0);
    return p0 + t * (p1 - p0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= GP.activeVoxels) { return; }

    let dim = P.sliceGridDim;
    let voxel = compactedVoxels[id];
    let gp = grid1DTo3D(voxel, dim);

    let dx = P.originSES.w;
    let origin = P.originSES.xyz;
    // p built from slice-LOCAL gridPos; world shift via offsetPos below (CUDA :768/:827).
    let p = origin + vec3<f32>(f32(gp.x), f32(gp.y), f32(gp.z)) * dx;
    let offsetPos = vec3<f32>(f32(P.offset.x), f32(P.offset.y), f32(P.offset.z)) * dx;

    var v: array<vec3<f32>, 8>;
    v[0] = p;
    v[1] = p + vec3<f32>(dx, 0.0, 0.0);
    v[2] = p + vec3<f32>(dx, dx, 0.0);
    v[3] = p + vec3<f32>(0.0, dx, 0.0);
    v[4] = p + vec3<f32>(0.0, 0.0, dx);
    v[5] = p + vec3<f32>(dx, 0.0, dx);
    v[6] = p + vec3<f32>(dx, dx, dx);
    v[7] = p + vec3<f32>(0.0, dx, dx);

    var field: array<f32, 8>;
    field[0] = gridValues[voxel];
    field[1] = gridValues[to1Du(gp + vec3<u32>(1u, 0u, 0u), dim)];
    field[2] = gridValues[to1Du(gp + vec3<u32>(1u, 1u, 0u), dim)];
    field[3] = gridValues[to1Du(gp + vec3<u32>(0u, 1u, 0u), dim)];
    field[4] = gridValues[to1Du(gp + vec3<u32>(0u, 0u, 1u), dim)];
    field[5] = gridValues[to1Du(gp + vec3<u32>(1u, 0u, 1u), dim)];
    field[6] = gridValues[to1Du(gp + vec3<u32>(1u, 1u, 1u), dim)];
    field[7] = gridValues[to1Du(gp + vec3<u32>(0u, 1u, 1u), dim)];

    let iso = 0.0;
    var cubeindex: i32 = 0;
    cubeindex = cubeindex + i32(field[0] < iso);
    cubeindex = cubeindex + i32(field[1] < iso) * 2;
    cubeindex = cubeindex + i32(field[2] < iso) * 4;
    cubeindex = cubeindex + i32(field[3] < iso) * 8;
    cubeindex = cubeindex + i32(field[4] < iso) * 16;
    cubeindex = cubeindex + i32(field[5] < iso) * 32;
    cubeindex = cubeindex + i32(field[6] < iso) * 64;
    cubeindex = cubeindex + i32(field[7] < iso) * 128;

    var vertlist: array<vec3<f32>, 12>;
    vertlist[0]  = vertexInterp(iso, v[0], v[1], field[0], field[1]);
    vertlist[1]  = vertexInterp(iso, v[1], v[2], field[1], field[2]);
    vertlist[2]  = vertexInterp(iso, v[2], v[3], field[2], field[3]);
    vertlist[3]  = vertexInterp(iso, v[3], v[0], field[3], field[0]);
    vertlist[4]  = vertexInterp(iso, v[4], v[5], field[4], field[5]);
    vertlist[5]  = vertexInterp(iso, v[5], v[6], field[5], field[6]);
    vertlist[6]  = vertexInterp(iso, v[6], v[7], field[6], field[7]);
    vertlist[7]  = vertexInterp(iso, v[7], v[4], field[7], field[4]);
    vertlist[8]  = vertexInterp(iso, v[0], v[4], field[0], field[4]);
    vertlist[9]  = vertexInterp(iso, v[1], v[5], field[1], field[5]);
    vertlist[10] = vertexInterp(iso, v[2], v[6], field[2], field[6]);
    vertlist[11] = vertexInterp(iso, v[3], v[7], field[3], field[7]);

    let numVerts = u32(nbTriTable[cubeindex]);
    let base = vertOffset[voxel];
    var i: u32 = 0u;
    loop {
        if (i >= numVerts) { break; }
        let index = base + i;
        let e0 = triTable[cubeindex * 16 + i32(i)];
        let e1 = triTable[cubeindex * 16 + i32(i) + 1];
        let e2 = triTable[cubeindex * 16 + i32(i) + 2];
        if (index < GP.maxVertsM3) {
            outVerts[index]     = vec4<f32>(vertlist[e0] + offsetPos, 0.0);
            outVerts[index + 1u] = vec4<f32>(vertlist[e1] + offsetPos, 0.0);
            outVerts[index + 2u] = vec4<f32>(vertlist[e2] + offsetPos, 0.0);
        }
        i = i + 3u;
    }
}
