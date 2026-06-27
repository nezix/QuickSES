# QuickSES cross-platform backends

QuickSES was CUDA-only. This `backends/` tree adds **portable compute backends** that reproduce the
CUDA mesh, so QuickSES runs beyond NVIDIA (macOS/Metal, Linux/Windows via Vulkan/D3D, Web, Quest).

The CUDA path stays in the repo root (`CudaSurf.cu` etc.) as the **NVIDIA fast reference** — it is
~5× faster than the portable backends at the raw surface-math kernels (benchmarked on an RTX A1000,
see the harness `resources/perf-benchmark.md`), so it remains the preferred path where CUDA exists.

## Backends

| Backend | Path | Runs on | Use |
|---------|------|---------|-----|
| **CUDA** (reference) | repo root | NVIDIA | fastest; the parity oracle. Unchanged algorithm + two banked perf wins. |
| **WGSL / WebGPU** | `backends/wgsl/` | Metal / Vulkan / D3D12 / **Web** (via wgpu/Dawn) | standalone cross-platform library; reaches web. |
| **Unity Compute Shader (HLSL)** | `backends/unity/` | every Unity target incl. WebGL/Quest, no native lib | the UnityMol plugin path (no FFI). |

## Parity (vs CUDA reference, validated)

All backends reproduce the CUDA QuickSES mesh. Validated bit-exact on vertices for the benchmark
ladder (1CRN/1UBQ/4HHB) at v=0.5, and on 1AON (58k atoms) including the **8-slab tiling** path:

| Structure | grid | slabs | verts | vs CUDA |
|-----------|------|-------|-------|---------|
| 1CRN | 74³ | 1 | 13,775 | 0 differing |
| 4HHB | 160³ | 1 | 146,565 | 0 differing |
| 1AON v=1.0 | 208³ | 1 | 463,264 | 0 differing |
| 1AON v=0.5 | 416³ | 8 | 1,951,253 | 8/1.95M differ (float seam, ≪ 0.01 Å) |

Faces differ by −1 per mesh: the CUDA CLI emits one degenerate triangle (`f i i i`) that QuickSES's
own `API_computeSES` path drops — the portable backends match the API-correct behavior.

## The `API_*` contract (preserved across backends)

The C ABI the consumers (UnityMol P/Invoke) depend on, defined in `CudaSurf.h`:

```c
void   API_computeSES(float resoSES, float3 *atomPos, float *atomRad, unsigned int N,
                      float3 *out_vertices, unsigned int *NVert,
                      int *out_triangles, unsigned int *NTri, int doSmoothing);
int   *API_getTriangles(bool invertTriangles);   // flat int[NTri], 3 per triangle
float3*API_getVertices();                          // float3[NVert]
int   *API_getAtomIdPerVert();                     // int[NVert]
void   API_freeMesh();
```
Results are exposed via getters + freed by `API_freeMesh` (caller owns the free). The WGSL backend
exposes the same C ABI when built as a shared lib (`cdylib`); the Unity backend fills a Unity `Mesh`
directly in-engine (no FFI) but mirrors the same vertex/triangle/atomId outputs.

## Build / run

- **WGSL** (`backends/wgsl/`): `cargo run --release -- -i <pdb> -o <out.obj> -v 0.5`. Runs on the
  system's default GPU backend (Metal on macOS, Vulkan on Linux, D3D12 on Windows). Prints a
  `TIMING …` line with the 6-phase breakdown. Cargo.lock is committed for reproducibility.
- **Unity** (`backends/unity/`): a Unity 2021.3 project. Headless run:
  `Unity -batchmode -quit -projectPath backends/unity -executeMethod QuickSES.QuickSESRunner.Run
  -pdb <pdb> -obj <out.obj> -reso 0.5 -logFile <log>` (NO `-nographics` — compute needs the GPU).
  In UnityMol it's driven via the plugin, not this CLI.

## Architecture (shared across portable backends)

The portable backends use the parity-safe split proven in the port: **GPU** does probeIntersection,
distanceFieldRefine, and the MC count/generate; **CPU** does neighbor bucketing, the exclusive-scan,
voxel compaction, and the vertex weld (snap→sort→unique→lower_bound). The weld is the parity-critical,
no-portable-primitive step (it sets the final vertex count) — keeping it on CPU locked bit-exact
counts. It is also the end-to-end perf bottleneck (~60% of wall time on large meshes), so moving it
onto the GPU is the highest-leverage future optimization.

Slab-tiling (for grids exceeding SLICE=300 cells/axis) replicates the CUDA halo + seam-dedup guards
exactly (`MarchingCubes.cu` border guards), so large structures tile without seam artifacts.
