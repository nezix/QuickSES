// distanceFieldRefine  (Kernels.cu:478)
// One thread per border cell. borderHashes[id] holds the SES hash of border cell id.
// offset = 0, sliceGrid == sesGrid.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> borderHashes: array<i32>; // compacted checkFill (notEmpty entries)
@group(0) @binding(2) var<storage, read_write> gridValues: array<f32>;

struct RefineParams {
    notEmptyCells: i32,
    _p0: i32,
    _p1: i32,
    _p2: i32,
};
@group(0) @binding(3) var<uniform> RP: RefineParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = i32(gid.x);
    if (id >= RP.notEmptyCells) { return; }

    let hash = borderHashes[id];
    let sliceDim = P.sliceGridDim;
    let ijk = unflatten1DTo3D(hash, sliceDim);

    // slice -> global cell id via reducedOffset (for world position)
    let ijkOffset = ijk + P.offset;
    let originSES = P.originSES.xyz;
    let dxSES = P.originSES.w;
    let spacePos3DCellSES = gridToSpace(ijkOffset, originSES, dxSES);

    var newresult: f32 = -dxSES;
    let idSESRangeToSearch = P.rangeSearchRefine; // ceil(PROBERADIUS/dxSES)
    let pme = PROBERADIUS - EPSILON;
    // Squared-distance hot loop (CUDA sqr_distance, Kernels.cu:372): compare squared
    // distances in the inner loop, take a single sqrt at the end (sqrt is monotonic so
    // min over d == sqrt(min over d^2)).
    var minDistSq: f32 = 100000.0 * 100000.0;

    // EXPANDING-SHELL branch-and-bound (bit-exact with the full (2R+1)^3 cube). The original
    // searches the whole cube for the nearest OUTSIDE cell. A cell at integer offset (x,y,z) with
    // Chebyshev radius r = max(|x|,|y|,|z|) has Euclidean distance >= dx*r, so once the current
    // shell's lower bound (r*dx) exceeds the best distance found so far, no farther cell can improve
    // minDist -> stop. Same visited set (shells union == cube), min idempotent => identical result,
    // reading only the shells up to the first hit + 1 instead of all (2R+1)^3 cells.
    // Squared-distance form: compare (r*dx)^2 against minDistSq, keeping the hot path sqrt-free.
    for (var r: i32 = 0; r <= idSESRangeToSearch; r = r + 1) {
        let rb = f32(r) * dxSES;
        if (rb * rb > minDistSq) { break; } // no farther shell can beat the current best
        for (var x: i32 = -r; x <= r; x = x + 1) {
            let cx = clampi(ijk.x + x, 0, sliceDim.x - 1);
            let ax = abs(x);
            for (var y: i32 = -r; y <= r; y = y + 1) {
                let cy = clampi(ijk.y + y, 0, sliceDim.y - 1);
                let ay = abs(y);
                // Only the SHELL at Chebyshev radius r: max(|x|,|y|,|z|) == r. If |x| or |y| is r,
                // z spans the full [-r,r]; otherwise z must be exactly +-r (2 cells).
                if (max(ax, ay) == r) {
                    for (var z: i32 = -r; z <= r; z = z + 1) {
                        let cz = clampi(ijk.z + z, 0, sliceDim.z - 1);
                        let cur = vec3<i32>(cx, cy, cz);
                        let curId = flatten3DTo1D(cur, sliceDim);
                        if (gridValues[curId] > pme) { // outside
                            let curOffset = cur + P.offset; // slice -> global
                            let spacePosSES = gridToSpace(curOffset, originSES, dxSES);
                            let dsq = sqr_distance(spacePosSES, spacePos3DCellSES);
                            minDistSq = min(dsq, minDistSq);
                        }
                    }
                } else {
                    for (var zs: i32 = 0; zs < 2; zs = zs + 1) {
                        if (r == 0 && zs == 1) { continue; } // r=0 is a single cell, don't double-count
                        var z: i32 = -r;
                        if (zs == 1) { z = r; }
                        let cz = clampi(ijk.z + z, 0, sliceDim.z - 1);
                        let cur = vec3<i32>(cx, cy, cz);
                        let curId = flatten3DTo1D(cur, sliceDim);
                        if (gridValues[curId] > pme) { // outside
                            let curOffset = cur + P.offset; // slice -> global
                            let spacePosSES = gridToSpace(curOffset, originSES, dxSES);
                            let dsq = sqr_distance(spacePosSES, spacePos3DCellSES);
                            minDistSq = min(dsq, minDistSq);
                        }
                    }
                }
            }
        }
    }
    if (minDistSq < (999.0 * 999.0)) {
        newresult = PROBERADIUS - sqrt(minDistSq);
    }
    gridValues[hash] = newresult;
}
