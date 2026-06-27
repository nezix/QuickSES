// QuickSES WGSL port — single-slab (offset = 0, sliceGrid == sesGrid).
// Faithful port of Kernels.cu / MarchingCubes.cu. All quirks of the CUDA code are
// reproduced deliberately (e.g. clamp uses gridDimNeighbor.x on all axes; probeIntersection
// builds the SES-cell world position from the neighbor-grid origin but the SES dx).
//
// Constants matching the CUDA source:
//   PROBERADIUS = 1.4, EPSILON = 0.001, EMPTYCELL = INT_MAX-1 = 2147483646

const PROBERADIUS: f32 = 1.4;
const EPSILON: f32 = 0.001;
const EMPTYCELL: i32 = 2147483646; // INT_MAX - 1

// Params shared by all kernels. std140-friendly layout (vec4 / scalars).
struct Params {
    // origin.xyz + dx in .w for neighbor grid
    originNeighbor: vec4<f32>,
    // origin.xyz + dx in .w for SES grid
    originSES: vec4<f32>,
    gridNeighborDim: vec3<i32>,
    _pad0: i32,
    gridSESDim: vec3<i32>,      // GLOBAL full-grid SES dim (gridSESDim in CUDA)
    _pad1: i32,
    sliceGridDim: vec3<i32>,    // slice-local processed slab edge (fullSliceGridSESDim)
    _pad1b: i32,
    offset: vec3<i32>,          // reducedOffset: slice read origin shifted back by halo
    _pad1c: i32,
    nAtoms: i32,
    rangeSearchRefine: i32,
    nSESCells: i32,             // slice cell count (sliceNbCellSES = sliceSize^3)
    _pad2: i32,
};

fn flatten3DTo1D(id: vec3<i32>, dim: vec3<i32>) -> i32 {
    return (dim.y * dim.z * id.x) + (dim.z * id.y) + id.z;
}

fn unflatten1DTo3D(index: i32, dim: vec3<i32>) -> vec3<i32> {
    let x = index / (dim.y * dim.z);
    let y = (index - x * dim.y * dim.z) / dim.z;
    let z = index - x * dim.y * dim.z - y * dim.z;
    return vec3<i32>(x, y, z);
}

fn spaceToGrid(pos: vec3<f32>, origin: vec3<f32>, dx: f32) -> vec3<i32> {
    let t = (pos - origin) / dx;
    return vec3<i32>(i32(t.x), i32(t.y), i32(t.z)); // C truncation toward zero
}

fn gridToSpace(cell: vec3<i32>, origin: vec3<f32>, dx: f32) -> vec3<f32> {
    return origin + vec3<f32>(f32(cell.x), f32(cell.y), f32(cell.z)) * dx;
}

fn fast_distance(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = a - b;
    return sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
}

// Squared distance — CUDA sqr_distance (Kernels.cu:156). Lets hot loops compare
// squared magnitudes and avoid the per-candidate sqrt.
fn sqr_distance(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = a - b;
    return d.x * d.x + d.y * d.y + d.z * d.z;
}

fn clampi(v: i32, a: i32, b: i32) -> i32 {
    return max(a, min(v, b));
}
