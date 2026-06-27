// QuickSES WGSL/WebGPU port — single-slab prototype.
// Faithful port of the CUDA SES mesher (CudaSurf.cu / Kernels.cu / MarchingCubes.cu).
// GPU (WGSL): probeIntersection, distanceFieldRefine, countVertexPerCell, generateTriangleVertices.
// CPU (Rust): PDB parse, grid setup, neighbor hash+sort+cellStartEnd, exclusive scans,
//             voxel compaction, vertex weld (snap+sort+unique+lower_bound), OBJ output.

mod tables;

use std::time::Instant;
use wgpu::util::DeviceExt;

const PROBERADIUS: f32 = 1.4;
const EPSILON: f32 = 0.001;
const EMPTYCELL: i32 = i32::MAX - 1; // 2147483646

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    origin_neighbor: [f32; 4],
    origin_ses: [f32; 4],
    grid_neighbor_dim: [i32; 3],
    _pad0: i32,
    grid_ses_dim: [i32; 3],   // GLOBAL full-grid SES dim
    _pad1: i32,
    slice_grid_dim: [i32; 3], // slice-local processed slab edge (fullSliceGridSESDim)
    _pad1b: i32,
    offset: [i32; 3],         // reducedOffset
    _pad1c: i32,
    n_atoms: i32,
    range_search_refine: i32,
    n_ses_cells: i32,         // slice cell count (sliceNbCellSES)
    _pad2: i32,
}

struct Atom {
    pos: [f32; 3],
    rad: f32,
}

// ---------- PDB parsing (matches cpdb: ATOM records only, element cols 77-78, vdW table) ----------
fn radius_for_element(e: char) -> f32 {
    match e {
        'O' => 1.52,
        'C' => 1.70,
        'N' => 1.55,
        'H' => 1.20,
        'S' => 1.80,
        'P' => 1.80,
        _ => 1.40, // 'X' default
    }
}

fn parse_pdb(path: &str) -> Vec<Atom> {
    let text = std::fs::read_to_string(path).expect("cannot read pdb");
    let mut atoms = Vec::new();
    for line in text.lines() {
        // cpdb matches "ATOM  " (6 chars) only; HETATM skipped.
        if !line.starts_with("ATOM  ") {
            // cpdb stops at ENDMDL / END
            if line.starts_with("ENDMDL") || line.starts_with("END") {
                break;
            }
            continue;
        }
        let b = line.as_bytes();
        if b.len() < 54 {
            continue;
        }
        // altLoc filter (default altLocFlag=1): keep 'A' or ' '
        let alt = if b.len() > 16 { b[16] as char } else { ' ' };
        if !(alt == 'A' || alt == ' ') {
            continue;
        }
        let getf = |s: usize, e: usize| -> f32 {
            let slice = &line[s..e.min(line.len())];
            slice.trim().parse::<f32>().unwrap_or(0.0)
        };
        let x = getf(30, 38);
        let y = getf(38, 46);
        let z = getf(46, 54);
        // element: cols 76..78 (0-indexed), trimmed; first char -> table.
        let mut elem = ' ';
        if b.len() >= 78 {
            let es = line[76..78].trim();
            if let Some(c) = es.chars().next() {
                elem = c;
            }
        }
        // cpdb extractStr trims leading spaces then uses element[0]; if column blank,
        // C reads garbage but in practice these PDBs have element columns populated.
        let rad = radius_for_element(elem);
        atoms.push(Atom { pos: [x, y, z], rad });
    }
    atoms
}

// ---------- grid helpers (match Kernels.cu) ----------
fn flatten3d(id: [i32; 3], dim: [i32; 3]) -> i32 {
    (dim[1] * dim[2] * id[0]) + (dim[2] * id[1]) + id[2]
}
fn space_to_grid(pos: [f32; 3], origin: [f32; 3], dx: f32) -> [i32; 3] {
    [
        ((pos[0] - origin[0]) / dx) as i32,
        ((pos[1] - origin[1]) / dx) as i32,
        ((pos[2] - origin[2]) / dx) as i32,
    ]
}

fn main() {
    // ----- CLI -----
    let args: Vec<String> = std::env::args().collect();
    let mut in_path = String::new();
    let mut out_path = String::new();
    let mut reso: f32 = 0.5;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-i" => { in_path = args[i + 1].clone(); i += 2; }
            "-o" => { out_path = args[i + 1].clone(); i += 2; }
            "-v" => { reso = args[i + 1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }
    if in_path.is_empty() {
        eprintln!("usage: wgsl-quickses -i <pdb> -o <obj> [-v 0.5]");
        std::process::exit(1);
    }
    if out_path.is_empty() {
        out_path = "out.obj".into();
    }

    let t_total = Instant::now();
    pollster::block_on(run(&in_path, &out_path, reso, t_total));
}

async fn run(in_path: &str, out_path: &str, reso: f32, t_total: Instant) {
    // ===== phase timers (all in ms, printed at the end) =====
    // t_setup: PDB parse + grid setup + neighbor bucketing + buffer create/upload.
    // t_gpu_compute: SUM of the 4 GPU dispatch regions (submit->poll), no CPU work inside
    //   the brackets. The CPU-scan design forces two mandatory mid-pipeline syncs (border
    //   compaction after probe; exclusive-scan after mc_count), so the GPU compute is the
    //   SUM of the four bracketed spans. Reported as such.
    // t_readback: GPU->host copies (the 3 staging readbacks).
    // t_weld: CPU snap+sort+unique+lower_bound + face build.
    // t_write: OBJ output. t_device_init: wgpu adapter/device acquisition (driver, not algo).
    let mut t_setup_ms: f64 = 0.0;
    let mut t_gpu_ms: f64 = 0.0;
    let mut t_readback_ms: f64 = 0.0;
    // t_host_scan: mid-pipeline CPU compaction the CPU-scan design forces between dispatches
    // (border filter+sort after probe; exclusive-scan + voxel compaction after mc_count).
    // Separate from t_weld so the six named phases stay clean.
    let mut t_host_ms: f64 = 0.0;
    let t_setup0 = Instant::now();

    // ---------- host grid setup (CudaSurf.cu:536-576) ----------
    let atoms = parse_pdb(in_path);
    let n = atoms.len();
    if n <= 1 {
        eprintln!("Failed to parse PDB or empty");
        std::process::exit(1);
    }

    let mut min_v = [f32::MAX; 3];
    let mut max_v = [f32::MIN; 3];
    let mut max_atom_rad = 0.0f32;
    for a in &atoms {
        for c in 0..3 {
            min_v[c] = min_v[c].min(a.pos[c]);
            max_v[c] = max_v[c].max(a.pos[c]);
        }
        max_atom_rad = max_atom_rad.max(a.rad);
    }

    let probe = PROBERADIUS;
    let max_dist = (max_v[0] - min_v[0])
        .max((max_v[1] - min_v[1]).max(max_v[2] - min_v[2]))
        + 2.0 * max_atom_rad
        + 4.0 * probe;

    let grid_reso_neighbor = probe + max_atom_rad;
    let origin_neighbor = [
        min_v[0] - max_atom_rad - probe,
        min_v[1] - max_atom_rad - probe,
        min_v[2] - max_atom_rad - probe,
    ];
    let grid_neighbor_size = (max_dist / grid_reso_neighbor).ceil() as i32;
    let grid_neighbor_dim = [grid_neighbor_size; 3];
    let grid_ses_size = (max_dist / reso).ceil() as i32;
    let grid_ses_dim = [grid_ses_size; 3];
    let n_neighbor_cells = (grid_neighbor_size as usize).pow(3);
    let range_search_refine = (probe / reso).ceil() as i32;

    // ---------- slab/slice model (CudaSurf.cu:629-635) ----------
    const SLICE: i32 = 300;
    let slice_small_size = SLICE.min(grid_ses_size); // stride per axis
    let slice_size = (SLICE + 2 * range_search_refine).min(grid_ses_size); // processed slab edge
    let slice_nb_cells = (slice_size as usize).pow(3); // sliceNbCellSES
    let slice_grid_dim = [slice_size; 3];
    // number of slabs along one axis (loop strides by slice_small_size)
    let n_slabs_axis = ((grid_ses_size + slice_small_size - 1) / slice_small_size).max(1);

    eprintln!("#atoms : {}", n);
    eprintln!(
        "Full size grid = {0} x {0} x {0}  (neighbor {1}^3, {2} cells)",
        grid_ses_size, grid_neighbor_size, (grid_ses_size as usize).pow(3)
    );
    eprintln!(
        "SLICE={SLICE} sliceSmall={slice_small_size} sliceSize={slice_size} halo(rangeSearchRefine)={range_search_refine} -> slabs={0}^3 = {1}",
        n_slabs_axis,
        n_slabs_axis * n_slabs_axis * n_slabs_axis
    );

    // ---------- neighbor-grid bucketing on CPU (hashAtoms + sort + sortCell) ----------
    // hash per atom
    let mut hash_index: Vec<(i32, i32)> = Vec::with_capacity(n);
    for (idx, a) in atoms.iter().enumerate() {
        let cell = space_to_grid(a.pos, origin_neighbor, grid_reso_neighbor);
        let hash = flatten3d(cell, grid_neighbor_dim);
        hash_index.push((hash, idx as i32));
    }
    // thrust::sort by hash (stable not required; CUDA uses a non-stable sort, but the
    // membership of cells is order-independent for our use). Sort by hash only.
    hash_index.sort_by_key(|x| x.0);

    // sorted atoms + cellStartEnd (sortCell)
    let mut sorted_atoms: Vec<[f32; 4]> = vec![[0.0; 4]; n];
    let mut cell_start_end: Vec<[i32; 2]> = vec![[EMPTYCELL, EMPTYCELL]; n_neighbor_cells];
    for (index, &(hash, id)) in hash_index.iter().enumerate() {
        let a = &atoms[id as usize];
        sorted_atoms[index] = [a.pos[0], a.pos[1], a.pos[2], a.rad];
        let hashm1 = if index != 0 { hash_index[index - 1].0 } else { hash };
        if index == 0 || hash != hashm1 {
            cell_start_end[hash as usize][0] = index as i32;
            if index > 0 {
                cell_start_end[hashm1 as usize][1] = index as i32;
            }
        }
        if index == n - 1 {
            cell_start_end[hash as usize][1] = (index + 1) as i32;
        }
    }

    // device init (wgpu adapter/device acquisition) is driver setup, not algorithm setup;
    // measure it on its own and exclude it from t_setup.
    t_setup_ms += t_setup0.elapsed().as_secs_f64() * 1000.0;
    let t_dev0 = Instant::now();

    // ---------- wgpu init ----------
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("no adapter");
    let info = adapter.get_info();
    eprintln!("ADAPTER: {:?} backend={:?}", info.name, info.backend);

    // Raise storage-buffer binding size limit to fit large grids (4HHB ~ 160^3 * 4 = ~16MB).
    let mut limits = wgpu::Limits::downlevel_defaults();
    let adapter_limits = adapter.limits();
    limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
    limits.max_storage_buffers_per_shader_stage = adapter_limits.max_storage_buffers_per_shader_stage;
    limits.max_buffer_size = adapter_limits.max_buffer_size;
    limits.max_compute_workgroups_per_dimension = adapter_limits.max_compute_workgroups_per_dimension;
    limits.max_compute_invocations_per_workgroup = adapter_limits.max_compute_invocations_per_workgroup;
    limits.max_compute_workgroup_size_x = adapter_limits.max_compute_workgroup_size_x;
    limits.max_compute_workgroup_size_y = adapter_limits.max_compute_workgroup_size_y;
    limits.max_compute_workgroup_size_z = adapter_limits.max_compute_workgroup_size_z;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: limits,
            },
            None,
        )
        .await
        .expect("device");

    let t_device_init_ms = t_dev0.elapsed().as_secs_f64() * 1000.0;
    // resume setup timing: param/buffer creation + uploads.
    let t_setup1 = Instant::now();

    // ---- shader header + module helper ----
    let header = include_str!("shaders/kernels.wgsl");
    let mk_module = |dev: &wgpu::Device, body: &str, label: &str| -> wgpu::ShaderModule {
        let src = format!("{}\n{}", header, body);
        dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        })
    };

    // ---- persistent neighbor-grid buffers (shared across all slabs) ----
    let cse_flat: Vec<i32> = cell_start_end.iter().flat_map(|c| [c[0], c[1]]).collect();
    let cse_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cellStartEnd"),
        contents: bytemuck::cast_slice(&cse_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let atoms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sortedAtoms"),
        contents: bytemuck::cast_slice(&sorted_atoms),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let nb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("nbTriTable"),
        contents: bytemuck::cast_slice(&tables::NB_TRI_TABLE),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let tri_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("triTable"),
        contents: bytemuck::cast_slice(&tables::TRI_TABLE),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // ---- per-slab grid buffers, sized to the slab (reused every slab) ----
    let slice_bytes = (slice_nb_cells * 4) as u64;
    let grid_vals = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gridValues"),
        size: slice_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let check_fill = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("checkFill"),
        size: slice_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let vpc_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertPerCell"),
        size: (slice_nb_cells * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ---- compute pipelines (persistent) ----
    let probe_mod = mk_module(&device, include_str!("shaders/probe.wgsl"), "probe");
    let probe_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("probe"), layout: None, module: &probe_mod, entry_point: "main",
        compilation_options: Default::default(),
    });
    let refine_mod = mk_module(&device, include_str!("shaders/refine.wgsl"), "refine");
    let refine_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("refine"), layout: None, module: &refine_mod, entry_point: "main",
        compilation_options: Default::default(),
    });
    let count_mod = mk_module(&device, include_str!("shaders/mc_count.wgsl"), "mc_count");
    let count_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mc_count"), layout: None, module: &count_mod, entry_point: "main",
        compilation_options: Default::default(),
    });
    let gen_mod = mk_module(&device, include_str!("shaders/mc_gen.wgsl"), "mc_gen");
    let gen_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mc_gen"), layout: None, module: &gen_mod, entry_point: "main",
        compilation_options: Default::default(),
    });

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct RefineParams { not_empty: i32, _p: [i32; 3] }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GenParams { active: u32, max_verts_m3: u32, _p: [u32; 2] }

    // setup (parse+grid+bucketing+buffer create/upload) ends here, before first slab.
    t_setup_ms += t_setup1.elapsed().as_secs_f64() * 1000.0;

    // global welded mesh, accumulated per slab (CUDA: per-slab weld then concat).
    let mut all_verts: Vec<[f32; 3]> = Vec::new();
    let mut all_faces: Vec<[i32; 3]> = Vec::new();
    let mut cumul_vert: i32 = 0; // vertex-index offset for triangle concat (writeToObj cumulVert)
    let mut total_border_cells: usize = 0;
    let mut slabs_with_mesh: usize = 0;

    // pre-zeroed init payloads, reused for every slab memset.
    let grid_init: Vec<f32> = vec![probe; slice_nb_cells];
    let zero_vpc: Vec<u8> = vec![0u8; slice_nb_cells * 8];

    // ---------- slab triple loop (CudaSurf.cu:653-754) ----------
    let mut i_off = 0i32;
    while i_off < grid_ses_size {
        let mut j_off = 0i32;
        while j_off < grid_ses_size {
            let mut k_off = 0i32;
            while k_off < grid_ses_size {
                // reducedOffset: shift read origin back by halo (CudaSurf.cu:671-673)
                let reduced = [
                    (i_off - range_search_refine).max(0),
                    (j_off - range_search_refine).max(0),
                    (k_off - range_search_refine).max(0),
                ];

                // per-slab params (sliceGridDim = fullSliceGridSESDim, offset = reducedOffset)
                let params = Params {
                    origin_neighbor: [origin_neighbor[0], origin_neighbor[1], origin_neighbor[2], grid_reso_neighbor],
                    origin_ses: [origin_neighbor[0], origin_neighbor[1], origin_neighbor[2], reso],
                    grid_neighbor_dim,
                    _pad0: 0,
                    grid_ses_dim,
                    _pad1: 0,
                    slice_grid_dim,
                    _pad1b: 0,
                    offset: reduced,
                    _pad1c: 0,
                    n_atoms: n as i32,
                    range_search_refine,
                    n_ses_cells: slice_nb_cells as i32,
                    _pad2: 0,
                };
                let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

                // memset slab grids (CudaSurf.cu:664-665)
                queue.write_buffer(&grid_vals, 0, bytemuck::cast_slice(&grid_init));
                // checkFill is fully overwritten by probe within the dispatched range, but the
                // dispatch is rounded up; init to EMPTYCELL so out-of-range cells filter out.
                {
                    let fill_init: Vec<i32> = vec![EMPTYCELL; slice_nb_cells];
                    queue.write_buffer(&check_fill, 0, bytemuck::cast_slice(&fill_init));
                }

                // ----- probeIntersection (dispatch over sliceSize, CudaSurf.cu:669) -----
                let probe_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None, layout: &probe_pipe.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: cse_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: atoms_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: grid_vals.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: check_fill.as_entire_binding() },
                    ],
                });
                {
                    let t_g0 = Instant::now();
                    let mut enc = device.create_command_encoder(&Default::default());
                    {
                        let mut cp = enc.begin_compute_pass(&Default::default());
                        cp.set_pipeline(&probe_pipe);
                        cp.set_bind_group(0, &probe_bind, &[]);
                        let g = ((slice_size as u32) + 7) / 8;
                        cp.dispatch_workgroups(g, g, g);
                    }
                    queue.submit(Some(enc.finish()));
                    device.poll(wgpu::Maintain::Wait);
                    t_gpu_ms += t_g0.elapsed().as_secs_f64() * 1000.0;
                }

                // ----- readback checkFill, compact border hashes on CPU -----
                let t_rb0 = Instant::now();
                let border_hashes = readback_i32(&device, &queue, &check_fill, slice_nb_cells);
                t_readback_ms += t_rb0.elapsed().as_secs_f64() * 1000.0;
                let t_h0 = Instant::now();
                let mut border: Vec<i32> = border_hashes.into_iter().filter(|&x| x != EMPTYCELL).collect();
                border.sort();
                let not_empty = border.len();
                t_host_ms += t_h0.elapsed().as_secs_f64() * 1000.0;
                total_border_cells += not_empty;

                if not_empty == 0 {
                    // empty slab — skip (CudaSurf.cu:693-697)
                    k_off += slice_small_size;
                    continue;
                }

                // ----- distanceFieldRefine -----
                let border_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("borderHashes"),
                    contents: bytemuck::cast_slice(&border),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let rp = RefineParams { not_empty: not_empty as i32, _p: [0; 3] };
                let rp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("refineParams"),
                    contents: bytemuck::bytes_of(&rp),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let refine_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None, layout: &refine_pipe.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: border_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: grid_vals.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: rp_buf.as_entire_binding() },
                    ],
                });
                {
                    let t_g0 = Instant::now();
                    let mut enc = device.create_command_encoder(&Default::default());
                    {
                        let mut cp = enc.begin_compute_pass(&Default::default());
                        cp.set_pipeline(&refine_pipe);
                        cp.set_bind_group(0, &refine_bind, &[]);
                        let g = ((not_empty as u32) + 255) / 256;
                        cp.dispatch_workgroups(g, 1, 1);
                    }
                    queue.submit(Some(enc.finish()));
                    device.poll(wgpu::Maintain::Wait);
                    t_gpu_ms += t_g0.elapsed().as_secs_f64() * 1000.0;
                }

                // ----- marching cubes: count (dispatch over sliceSize) -----
                queue.write_buffer(&vpc_buf, 0, &zero_vpc);
                let count_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None, layout: &count_pipe.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: grid_vals.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: vpc_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: nb_buf.as_entire_binding() },
                    ],
                });
                {
                    let t_g0 = Instant::now();
                    let mut enc = device.create_command_encoder(&Default::default());
                    {
                        let mut cp = enc.begin_compute_pass(&Default::default());
                        cp.set_pipeline(&count_pipe);
                        cp.set_bind_group(0, &count_bind, &[]);
                        let g = ((slice_size as u32) + 7) / 8;
                        cp.dispatch_workgroups(g, g, g);
                    }
                    queue.submit(Some(enc.finish()));
                    device.poll(wgpu::Maintain::Wait);
                    t_gpu_ms += t_g0.elapsed().as_secs_f64() * 1000.0;
                }

                // ----- readback vertPerCell, exclusive scan + compaction on CPU -----
                let t_rb1 = Instant::now();
                let vpc_raw = readback_u32(&device, &queue, &vpc_buf, slice_nb_cells * 2);
                t_readback_ms += t_rb1.elapsed().as_secs_f64() * 1000.0;
                let t_h1 = Instant::now();
                let mut vert_offset = vec![0u32; slice_nb_cells];
                let mut compacted: Vec<u32> = Vec::new();
                let mut acc_v: u32 = 0;
                for cell in 0..slice_nb_cells {
                    let nv = vpc_raw[2 * cell];
                    let occ = vpc_raw[2 * cell + 1];
                    vert_offset[cell] = acc_v;
                    acc_v = acc_v.wrapping_add(nv);
                    if occ > 0 {
                        compacted.push(cell as u32);
                    }
                }
                let total_verts = acc_v as usize;
                let active_voxels = compacted.len();
                t_host_ms += t_h1.elapsed().as_secs_f64() * 1000.0;
                if total_verts == 0 || active_voxels == 0 {
                    k_off += slice_small_size;
                    continue;
                }

                // ----- marching cubes: generate -----
                let compacted_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("compactedVoxels"),
                    contents: bytemuck::cast_slice(&compacted),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let voffset_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("vertOffset"),
                    contents: bytemuck::cast_slice(&vert_offset),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let out_verts_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("outVerts"),
                    size: (total_verts * 16) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let gp = GenParams {
                    active: active_voxels as u32,
                    max_verts_m3: (total_verts as u32).wrapping_sub(3),
                    _p: [0; 2],
                };
                let gp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("genParams"),
                    contents: bytemuck::bytes_of(&gp),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let gen_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None, layout: &gen_pipe.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: grid_vals.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: compacted_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: voffset_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: tri_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: nb_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: out_verts_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 7, resource: gp_buf.as_entire_binding() },
                    ],
                });
                {
                    let t_g0 = Instant::now();
                    let mut enc = device.create_command_encoder(&Default::default());
                    {
                        let mut cp = enc.begin_compute_pass(&Default::default());
                        cp.set_pipeline(&gen_pipe);
                        cp.set_bind_group(0, &gen_bind, &[]);
                        let g = ((active_voxels as u32) + 255) / 256;
                        cp.dispatch_workgroups(g, 1, 1);
                    }
                    queue.submit(Some(enc.finish()));
                    device.poll(wgpu::Maintain::Wait);
                    t_gpu_ms += t_g0.elapsed().as_secs_f64() * 1000.0;
                }

                // ----- readback vertices, per-slab weld, concat -----
                let t_rb2 = Instant::now();
                let raw = readback_f32(&device, &queue, &out_verts_buf, total_verts * 4);
                t_readback_ms += t_rb2.elapsed().as_secs_f64() * 1000.0;
                let t_weld0 = Instant::now();
                let mut verts: Vec<[f32; 3]> = Vec::with_capacity(total_verts);
                for v in 0..total_verts {
                    verts.push([raw[4 * v], raw[4 * v + 1], raw[4 * v + 2]]);
                }

                // per-slab weld (CudaSurf.cu:425-499)
                let (welded_verts, tri_index) = weld(&verts);
                let n_triangles = total_verts / 3; // Ntriangles = totalVerts/3

                // concat: triangle indices offset by cumulative welded-vertex count
                // (writeToObj cumulVert, CudaSurf.cu:363). Degenerate triangles dropped
                // (API aggregation, CudaSurf.cu:823).
                for t in 0..n_triangles {
                    let a = tri_index[t * 3];
                    let b = tri_index[t * 3 + 1];
                    let c = tri_index[t * 3 + 2];
                    if a != b && b != c && a != c {
                        all_faces.push([a + cumul_vert, b + cumul_vert, c + cumul_vert]);
                    }
                }
                cumul_vert += welded_verts.len() as i32;
                all_verts.extend_from_slice(&welded_verts);
                slabs_with_mesh += 1;
                t_host_ms += t_weld0.elapsed().as_secs_f64() * 1000.0;

                k_off += slice_small_size;
            }
            j_off += slice_small_size;
        }
        i_off += slice_small_size;
    }

    eprintln!("border cells (all slabs) = {}  slabs with mesh = {}", total_border_cells, slabs_with_mesh);

    if all_verts.is_empty() {
        eprintln!("Empty surface");
        write_obj(out_path, &[], &[]);
        return;
    }

    let t_write0 = Instant::now();
    write_obj(out_path, &all_verts, &all_faces);
    let t_write_ms = t_write0.elapsed().as_secs_f64() * 1000.0;

    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    let id = std::path::Path::new(in_path).file_stem().unwrap().to_string_lossy();

    // machine-parseable timing line (one per run; median computed by the harness).
    // weld time is folded into host_scan here (per-slab weld interleaves with the loop).
    println!(
        "TIMING {id} setup={t_setup_ms:.2} device_init={t_device_init_ms:.2} gpu_compute={t_gpu_ms:.2} readback={t_readback_ms:.2} host_scan={t_host_ms:.2} weld=0.00 write={t_write_ms:.2} end_to_end={total_ms:.2} verts={} faces={}",
        all_verts.len(),
        all_faces.len(),
    );
}

// Vertex weld: snap to EPSILON grid, sort lexicographically, unique, lower_bound.
// Returns (unique welded verts, per-original-vertex unique index).
fn weld(verts: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<i32>) {
    let snap = |x: f32| -> f32 { (x / EPSILON).round() * EPSILON };
    let snapped: Vec<[f32; 3]> = verts
        .iter()
        .map(|v| [snap(v[0]), snap(v[1]), snap(v[2])])
        .collect();

    // sorted unique (lexicographic by x,y,z — matches thrust tuple<f32,f32,f32> default <)
    let mut uniq = snapped.clone();
    uniq.sort_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap()
            .then(a[1].partial_cmp(&b[1]).unwrap())
            .then(a[2].partial_cmp(&b[2]).unwrap())
    });
    uniq.dedup();

    // lower_bound of each original snapped vertex within uniq -> triangle index
    let tri_index: Vec<i32> = snapped
        .iter()
        .map(|v| {
            // binary search for first element >= v (lower_bound)
            let mut lo = 0usize;
            let mut hi = uniq.len();
            while lo < hi {
                let mid = (lo + hi) / 2;
                if cmp_vec(&uniq[mid], v) == std::cmp::Ordering::Less {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo as i32
        })
        .collect();

    (uniq, tri_index)
}

fn cmp_vec(a: &[f32; 3], b: &[f32; 3]) -> std::cmp::Ordering {
    a[0].partial_cmp(&b[0])
        .unwrap()
        .then(a[1].partial_cmp(&b[1]).unwrap())
        .then(a[2].partial_cmp(&b[2]).unwrap())
}

// OBJ winding: CUDA emits "f y+1 x+1 z+1" (CudaSurf.cu:322/363).
fn write_obj(path: &str, verts: &[[f32; 3]], faces: &[[i32; 3]]) {
    use std::fmt::Write;
    let mut s = String::with_capacity(verts.len() * 24 + faces.len() * 24);
    for v in verts {
        writeln!(s, "v {:.3} {:.3} {:.3}", v[0], v[1], v[2]).unwrap();
    }
    s.push('\n');
    for f in faces {
        writeln!(s, "f {} {} {}", f[1] + 1, f[0] + 1, f[2] + 1).unwrap();
    }
    std::fs::write(path, s).unwrap();
}

// ---------- readback helpers ----------
fn readback_bytes(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, n_elems: usize, elem_size: usize) -> Vec<u8> {
    let size = (n_elems * elem_size) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
    queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range().to_vec();
    data
}
fn readback_i32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, n: usize) -> Vec<i32> {
    bytemuck::cast_slice(&readback_bytes(device, queue, buf, n, 4)).to_vec()
}
fn readback_u32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, n: usize) -> Vec<u32> {
    bytemuck::cast_slice(&readback_bytes(device, queue, buf, n, 4)).to_vec()
}
fn readback_f32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
    bytemuck::cast_slice(&readback_bytes(device, queue, buf, n, 4)).to_vec()
}
