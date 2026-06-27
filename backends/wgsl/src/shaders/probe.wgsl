// probeIntersection + computeInOrOut  (Kernels.cu:344, :394)
// Classifies each SES cell out / in / border. offset = (0,0,0), sliceGrid == sesGrid.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> cellStartEnd: array<vec2<i32>>; // per neighbor cell {start,end}
@group(0) @binding(2) var<storage, read> sortedAtoms: array<vec4<f32>>;  // {x,y,z,r}
@group(0) @binding(3) var<storage, read_write> gridValues: array<f32>;
@group(0) @binding(4) var<storage, read_write> checkFill: array<i32>;

fn computeInOrOut(id3DNeigh: vec3<i32>, spacePosSES: vec3<f32>, dx: f32) -> f32 {
    var result: f32 = PROBERADIUS;
    var nearProbe: bool = false;
    let nd = P.gridNeighborDim;
    for (var x: i32 = -1; x <= 1; x = x + 1) {
        let cx = clampi(id3DNeigh.x + x, 0, nd.x - 1);
        for (var y: i32 = -1; y <= 1; y = y + 1) {
            // CUDA quirk: uses gridDimNeighbor.x for the y clamp too
            let cy = clampi(id3DNeigh.y + y, 0, nd.x - 1);
            for (var z: i32 = -1; z <= 1; z = z + 1) {
                let cz = clampi(id3DNeigh.z + z, 0, nd.x - 1);
                let neighcellhash = flatten3DTo1D(vec3<i32>(cx, cy, cz), nd);
                let se = cellStartEnd[neighcellhash];
                let idStart = se.x;
                let idStop = se.y;
                if (idStart < EMPTYCELL) {
                    for (var idx: i32 = idStart; idx < idStop; idx = idx + 1) {
                        let a = sortedAtoms[idx];
                        let rad = a.w;
                        let pos = vec3<f32>(a.x, a.y, a.z);
                        // Squared-distance compare: rad-dx and PROBERADIUS+rad are both
                        // non-negative here, so d < t  <=>  d^2 < t^2. Avoids per-atom sqrt.
                        let dsq = sqr_distance(pos, spacePosSES);
                        let tIn = rad - dx;
                        if (dsq < tIn * tIn) {
                            return -dx;
                        }
                        let tProbe = PROBERADIUS + rad;
                        if (dsq < tProbe * tProbe) {
                            nearProbe = true;
                        }
                    }
                }
            }
        }
    }
    if (nearProbe) {
        result = 0.0;
    }
    return result;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    let k = i32(gid.z);
    let sliceDim = P.sliceGridDim;

    if (i >= sliceDim.x - 1) { return; }
    if (j >= sliceDim.y - 1) { return; }
    if (k >= sliceDim.z - 1) { return; }

    let ijk = vec3<i32>(i, j, k);
    let hash = flatten3DTo1D(ijk, sliceDim);

    // slice -> global cell id via reducedOffset
    let ijkOffset = ijk + P.offset;
    if (ijkOffset.x >= P.gridSESDim.x - 1) { return; }
    if (ijkOffset.y >= P.gridSESDim.y - 1) { return; }
    if (ijkOffset.z >= P.gridSESDim.z - 1) { return; }

    let originNeighbor = P.originNeighbor.xyz;
    let dxNeighbor = P.originNeighbor.w;
    let dxSES = P.originSES.w;

    // CUDA quirk: SES-cell world position built from neighbor origin but SES dx
    let spacePos3DCellSES = gridToSpace(ijkOffset, originNeighbor, dxSES);
    let gridPos3DCellNeighbor = spaceToGrid(spacePos3DCellSES, originNeighbor, dxNeighbor);

    let result = computeInOrOut(gridPos3DCellNeighbor, spacePos3DCellSES, dxSES);

    var fill: i32 = EMPTYCELL;
    if (abs(result) < EPSILON) {
        fill = hash;
    }
    checkFill[hash] = fill;
    gridValues[hash] = result;
}
