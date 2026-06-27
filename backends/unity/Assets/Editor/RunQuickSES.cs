// QuickSES Unity host (C#) — mirrors wgsl-quickses/src/main.rs stage-for-stage.
// CPU: PDB parse, grid setup, neighbor hash+sort+cellStartEnd, exclusive scan, compaction,
//      vertex weld (snap+sort+unique+lower_bound), OBJ output.
// GPU (HLSL compute): probeIntersection, distanceFieldRefine, countVertexPerCell, generateTriangleVertices.
//
// Drive headless:
//   Unity -batchmode -quit -projectPath <proj> -executeMethod QuickSES.QuickSESRunner.Run -logFile <log>
// (do NOT use -nographics — compute shaders need the Metal GPU.)

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace QuickSES
{
    public static class QuickSESRunner
    {
        const float PROBERADIUS = 1.4f;
        const float EPSILON = 0.001f;
        const int   EMPTYCELL = 2147483646; // INT_MAX - 1

        struct Atom { public float x, y, z, rad; }

        // Blittable struct mirroring the cbuffer Params layout (96 bytes).
        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
        struct ParamsStruct
        {
            public float onx, ony, onz, ndx;   // originNeighbor
            public float osx, osy, osz, sdx;    // originSES
            public int gnx, gny, gnz, gnp;      // gridNeighborDim int4
            public int gsx, gsy, gsz, gsp;      // gridSESDim int4
            public int nAtoms, rangeSearchRefine, nSESCells, notEmptyCells;
            public int activeVoxels, maxVertsM3, pad0, pad1;
        }

        static float RadiusForElement(char e)
        {
            switch (e)
            {
                case 'O': return 1.52f;
                case 'C': return 1.70f;
                case 'N': return 1.55f;
                case 'H': return 1.20f;
                case 'S': return 1.80f;
                case 'P': return 1.80f;
                default:  return 1.40f; // 'X'
            }
        }

        static float ParseF(string line, int s, int e)
        {
            int end = Math.Min(e, line.Length);
            if (s >= end) return 0f;
            string slice = line.Substring(s, end - s).Trim();
            float v;
            if (float.TryParse(slice, NumberStyles.Float, CultureInfo.InvariantCulture, out v)) return v;
            return 0f;
        }

        static List<Atom> ParsePdb(string path)
        {
            var atoms = new List<Atom>();
            foreach (var line in File.ReadAllLines(path))
            {
                if (!line.StartsWith("ATOM  "))
                {
                    if (line.StartsWith("ENDMDL") || line.StartsWith("END")) break;
                    continue;
                }
                if (line.Length < 54) continue;
                char alt = line.Length > 16 ? line[16] : ' ';
                if (!(alt == 'A' || alt == ' ')) continue;

                float x = ParseF(line, 30, 38);
                float y = ParseF(line, 38, 46);
                float z = ParseF(line, 46, 54);
                char elem = ' ';
                if (line.Length >= 78)
                {
                    string es = line.Substring(76, 2).Trim();
                    if (es.Length > 0) elem = es[0];
                }
                float rad = RadiusForElement(elem);
                atoms.Add(new Atom { x = x, y = y, z = z, rad = rad });
            }
            return atoms;
        }

        static int Flatten3D(int ix, int iy, int iz, int dx, int dy, int dz)
        {
            return (dy * dz * ix) + (dz * iy) + iz;
        }

        // ---- entry point ----
        public static void Run()
        {
            try
            {
                string[] argv = Environment.GetCommandLineArgs();
                string inArg = null, outArg = null;
                float reso = 0.5f;
                for (int i = 0; i < argv.Length; i++)
                {
                    if (argv[i] == "-pdb" && i + 1 < argv.Length) inArg = argv[i + 1];
                    if (argv[i] == "-obj" && i + 1 < argv.Length) outArg = argv[i + 1];
                    if (argv[i] == "-reso" && i + 1 < argv.Length) float.TryParse(argv[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out reso);
                }

                LogSysInfo();

                if (inArg != null)
                {
                    RunOne(inArg, outArg ?? "out.obj", reso);
                }
                else
                {
                    // default: the 3 fixtures
                    string fx = "/opt/src/ai/prod/quickses-x-claudia/fixtures/pdb";
                    string outDir = Path.Combine(Application.dataPath, "..", "out");
                    Directory.CreateDirectory(outDir);
                    foreach (var name in new[] { "1CRN", "1UBQ", "4HHB" })
                    {
                        RunOne(Path.Combine(fx, name + ".pdb"), Path.Combine(outDir, name + ".obj"), reso);
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogError("[QuickSES] FATAL: " + ex);
                EditorApplication.Exit(2);
                return;
            }
            EditorApplication.Exit(0);
        }

        static void LogSysInfo()
        {
            Debug.Log("[QuickSES] graphicsDeviceType=" + SystemInfo.graphicsDeviceType
                + " device=" + SystemInfo.graphicsDeviceName
                + " supportsComputeShaders=" + SystemInfo.supportsComputeShaders);
        }

        static void RunOne(string inPath, string outPath, float reso)
        {
            if (!SystemInfo.supportsComputeShaders)
            {
                Debug.LogError("[QuickSES] Compute shaders NOT supported on this device. Aborting.");
                throw new Exception("no compute support");
            }

            var atoms = ParsePdb(inPath);
            int n = atoms.Count;
            Debug.Log("[QuickSES] " + Path.GetFileNameWithoutExtension(inPath) + " #atoms=" + n);
            if (n <= 1) { Debug.LogError("[QuickSES] empty/failed parse"); return; }

            // ---- grid setup (CudaSurf.cu:536-576) ----
            float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;
            float maxAtomRad = 0f;
            foreach (var a in atoms)
            {
                minX = Mathf.Min(minX, a.x); minY = Mathf.Min(minY, a.y); minZ = Mathf.Min(minZ, a.z);
                maxX = Mathf.Max(maxX, a.x); maxY = Mathf.Max(maxY, a.y); maxZ = Mathf.Max(maxZ, a.z);
                maxAtomRad = Mathf.Max(maxAtomRad, a.rad);
            }
            float probe = PROBERADIUS;
            float maxDist = Mathf.Max(maxX - minX, Mathf.Max(maxY - minY, maxZ - minZ))
                            + 2.0f * maxAtomRad + 4.0f * probe;

            float gridResoNeighbor = probe + maxAtomRad;
            float oNX = minX - maxAtomRad - probe;
            float oNY = minY - maxAtomRad - probe;
            float oNZ = minZ - maxAtomRad - probe;
            int gridNeighborSize = (int)Mathf.Ceil(maxDist / gridResoNeighbor);
            int gridSESSize = (int)Mathf.Ceil(maxDist / reso);
            int nNeighborCells = gridNeighborSize * gridNeighborSize * gridNeighborSize;
            int nSESCells = gridSESSize * gridSESSize * gridSESSize;
            int rangeSearchRefine = (int)Mathf.Ceil(probe / reso);

            Debug.Log("[QuickSES] grid SES=" + gridSESSize + "^3 neighbor=" + gridNeighborSize
                + "^3 nSESCells=" + nSESCells);

            // ---- neighbor bucketing on CPU (hashAtoms + sort + sortCell) ----
            var hashIndex = new List<KeyValuePair<int, int>>(n);
            for (int idx = 0; idx < n; idx++)
            {
                var a = atoms[idx];
                int cx = (int)((a.x - oNX) / gridResoNeighbor);
                int cy = (int)((a.y - oNY) / gridResoNeighbor);
                int cz = (int)((a.z - oNZ) / gridResoNeighbor);
                int hash = Flatten3D(cx, cy, cz, gridNeighborSize, gridNeighborSize, gridNeighborSize);
                hashIndex.Add(new KeyValuePair<int, int>(hash, idx));
            }
            // sort by hash only (stable not required; cell membership order-independent)
            hashIndex.Sort((p, q) => p.Key.CompareTo(q.Key));

            var sortedAtoms = new float[n * 4];
            var cellStartEnd = new int[nNeighborCells * 2];
            for (int c = 0; c < nNeighborCells; c++) { cellStartEnd[2 * c] = EMPTYCELL; cellStartEnd[2 * c + 1] = EMPTYCELL; }
            for (int index = 0; index < n; index++)
            {
                int hash = hashIndex[index].Key;
                int id = hashIndex[index].Value;
                var a = atoms[id];
                sortedAtoms[4 * index] = a.x; sortedAtoms[4 * index + 1] = a.y;
                sortedAtoms[4 * index + 2] = a.z; sortedAtoms[4 * index + 3] = a.rad;
                int hashm1 = index != 0 ? hashIndex[index - 1].Key : hash;
                if (index == 0 || hash != hashm1)
                {
                    cellStartEnd[2 * hash] = index;
                    if (index > 0) cellStartEnd[2 * hashm1 + 1] = index;
                }
                if (index == n - 1) cellStartEnd[2 * hash + 1] = index + 1;
            }

            // ---- compute ----
            var cs = AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/QuickSES.compute");
            if (cs == null) { Debug.LogError("[QuickSES] cannot load compute shader asset"); throw new Exception("no shader"); }

            int kProbe = cs.FindKernel("probeIntersection");
            int kRefine = cs.FindKernel("distanceFieldRefine");
            int kCount = cs.FindKernel("countVertexPerCell");
            int kGen = cs.FindKernel("generateTriangleVertices");

            // ---- Params constant buffer (matches cbuffer layout in QuickSES.compute) ----
            // float4 originNeighbor; float4 originSES; int4 gridNeighborDim; int4 gridSESDim;
            // int nAtoms; int rangeSearchRefine; int nSESCells; int notEmptyCells;
            // int activeVoxels; int maxVertsM3; int _pad0; int _pad1;
            // total = 4*4 + 4*4 + 4*4 + 4*4 + 8*4 = 96 bytes
            const int PARAMS_BYTES = 96;
            var paramsBuf = new ComputeBuffer(1, PARAMS_BYTES, ComputeBufferType.Constant);

            Action<int, int, int> setParams = (notEmpty, active, maxV3) =>
            {
                var ps = new ParamsStruct
                {
                    onx = oNX, ony = oNY, onz = oNZ, ndx = gridResoNeighbor,
                    // originSES: origin = neighbor origin, .w = reso — matches main.rs
                    osx = oNX, osy = oNY, osz = oNZ, sdx = reso,
                    gnx = gridNeighborSize, gny = gridNeighborSize, gnz = gridNeighborSize, gnp = 0,
                    gsx = gridSESSize, gsy = gridSESSize, gsz = gridSESSize, gsp = 0,
                    nAtoms = n, rangeSearchRefine = rangeSearchRefine, nSESCells = nSESCells, notEmptyCells = notEmpty,
                    activeVoxels = active, maxVertsM3 = maxV3, pad0 = 0, pad1 = 0,
                };
                paramsBuf.SetData(new[] { ps });
            };
            setParams(0, 0, 0);

            // ---- buffers ----
            var cseBuf = new ComputeBuffer(nNeighborCells, 8); // int2
            cseBuf.SetData(cellStartEnd);
            var atomsBuf = new ComputeBuffer(n, 16);           // float4
            atomsBuf.SetData(sortedAtoms);

            var gridVals = new ComputeBuffer(nSESCells, 4);
            var initGrid = new float[nSESCells];
            for (int i = 0; i < nSESCells; i++) initGrid[i] = probe;
            gridVals.SetData(initGrid);
            var checkFill = new ComputeBuffer(nSESCells, 4);

            int g = (gridSESSize + 7) / 8;

            // ---- probeIntersection ----
            cs.SetConstantBuffer("Params", paramsBuf, 0, PARAMS_BYTES);
            cs.SetBuffer(kProbe, "cellStartEnd", cseBuf);
            cs.SetBuffer(kProbe, "sortedAtoms", atomsBuf);
            cs.SetBuffer(kProbe, "gridValues", gridVals);
            cs.SetBuffer(kProbe, "checkFill", checkFill);
            cs.Dispatch(kProbe, g, g, g);

            // ---- readback checkFill, compact border hashes on CPU ----
            var checkFillArr = new int[nSESCells];
            checkFill.GetData(checkFillArr);
            var border = new List<int>();
            for (int i = 0; i < nSESCells; i++) if (checkFillArr[i] != EMPTYCELL) border.Add(checkFillArr[i]);
            border.Sort();
            int notEmpty = border.Count;
            Debug.Log("[QuickSES]   border cells=" + notEmpty);

            if (notEmpty == 0)
            {
                Debug.Log("[QuickSES]   empty surface");
                WriteObj(outPath, new List<float[]>(), new List<int[]>());
                ReleaseAll(paramsBuf, cseBuf, atomsBuf, gridVals, checkFill);
                Debug.Log("[QuickSES] RESULT " + Path.GetFileNameWithoutExtension(inPath) + " verts=0 faces=0");
                return;
            }

            // ---- distanceFieldRefine ----
            setParams(notEmpty, 0, 0);
            cs.SetConstantBuffer("Params", paramsBuf, 0, PARAMS_BYTES);
            var borderBuf = new ComputeBuffer(notEmpty, 4);
            borderBuf.SetData(border.ToArray());
            cs.SetBuffer(kRefine, "borderHashes", borderBuf);
            cs.SetBuffer(kRefine, "gridValues", gridVals);
            int gr = (notEmpty + 255) / 256;
            cs.Dispatch(kRefine, gr, 1, 1);

            // ---- MC count ----
            var nbBuf = new ComputeBuffer(256, 4);
            nbBuf.SetData(MCTables.NbTriTable);
            var vpcBuf = new ComputeBuffer(nSESCells, 8); // uint2
            vpcBuf.SetData(new uint[nSESCells * 2]);      // zeroed
            cs.SetConstantBuffer("Params", paramsBuf, 0, PARAMS_BYTES);
            cs.SetBuffer(kCount, "gridValues", gridVals);
            cs.SetBuffer(kCount, "vertPerCell", vpcBuf);
            cs.SetBuffer(kCount, "nbTriTable", nbBuf);
            cs.Dispatch(kCount, g, g, g);

            // ---- readback vertPerCell, exclusive scan + compaction on CPU ----
            var vpcRaw = new uint[nSESCells * 2];
            vpcBuf.GetData(vpcRaw);
            var vertOffset = new uint[nSESCells];
            var compacted = new List<uint>();
            uint accV = 0;
            for (int cell = 0; cell < nSESCells; cell++)
            {
                uint nv = vpcRaw[2 * cell];
                uint occ = vpcRaw[2 * cell + 1];
                vertOffset[cell] = accV;
                accV = unchecked(accV + nv);
                if (occ > 0) compacted.Add((uint)cell);
            }
            int totalVerts = (int)accV;
            int activeVoxels = compacted.Count;
            if (totalVerts == 0 || activeVoxels == 0)
            {
                Debug.Log("[QuickSES]   no vertices");
                WriteObj(outPath, new List<float[]>(), new List<int[]>());
                ReleaseAll(paramsBuf, cseBuf, atomsBuf, gridVals, checkFill, borderBuf, nbBuf, vpcBuf);
                Debug.Log("[QuickSES] RESULT " + Path.GetFileNameWithoutExtension(inPath) + " verts=0 faces=0");
                return;
            }

            // ---- MC generate ----
            setParams(notEmpty, activeVoxels, totalVerts - 3);
            cs.SetConstantBuffer("Params", paramsBuf, 0, PARAMS_BYTES);
            var triBuf = new ComputeBuffer(MCTables.TriTable.Length, 4);
            triBuf.SetData(MCTables.TriTable);
            var compactedBuf = new ComputeBuffer(activeVoxels, 4);
            compactedBuf.SetData(compacted.ToArray());
            var voffBuf = new ComputeBuffer(nSESCells, 4);
            voffBuf.SetData(vertOffset);
            var outVertsBuf = new ComputeBuffer(totalVerts, 16); // float4
            cs.SetBuffer(kGen, "gridValues", gridVals);
            cs.SetBuffer(kGen, "compactedVoxels", compactedBuf);
            cs.SetBuffer(kGen, "vertOffset", voffBuf);
            cs.SetBuffer(kGen, "triTable", triBuf);
            cs.SetBuffer(kGen, "nbTriTable", nbBuf);
            cs.SetBuffer(kGen, "outVerts", outVertsBuf);
            int gg = (activeVoxels + 255) / 256;
            cs.Dispatch(kGen, gg, 1, 1);

            // ---- readback vertices ----
            var raw = new float[totalVerts * 4];
            outVertsBuf.GetData(raw);
            var verts = new List<float[]>(totalVerts);
            for (int v = 0; v < totalVerts; v++)
                verts.Add(new[] { raw[4 * v], raw[4 * v + 1], raw[4 * v + 2] });

            // ---- vertex weld on CPU (CudaSurf.cu:425-499) ----
            List<float[]> weldedVerts;
            int[] triIndex;
            Weld(verts, out weldedVerts, out triIndex);
            int nTriangles = totalVerts / 3;

            var faces = new List<int[]>(nTriangles);
            for (int t = 0; t < nTriangles; t++)
            {
                int a = triIndex[t * 3];
                int b = triIndex[t * 3 + 1];
                int c = triIndex[t * 3 + 2];
                if (a != b && b != c && a != c) faces.Add(new[] { a, b, c });
            }

            WriteObj(outPath, weldedVerts, faces);
            Debug.Log("[QuickSES] RESULT " + Path.GetFileNameWithoutExtension(inPath)
                + " verts=" + weldedVerts.Count + " faces=" + faces.Count
                + " (raw_mc_verts=" + totalVerts + " active_voxels=" + activeVoxels + " border=" + notEmpty + ")");

            ReleaseAll(paramsBuf, cseBuf, atomsBuf, gridVals, checkFill, borderBuf, nbBuf, vpcBuf,
                       triBuf, compactedBuf, voffBuf, outVertsBuf);
        }

        static void ReleaseAll(params ComputeBuffer[] bufs)
        {
            foreach (var b in bufs) if (b != null) b.Release();
        }

        // Vertex weld: snap to EPSILON grid, sort lexicographically, unique, lower_bound.
        static void Weld(List<float[]> verts, out List<float[]> uniq, out int[] triIndex)
        {
            Func<float, float> snap = (x) => Mathf.Round(x / EPSILON) * EPSILON;
            var snapped = new float[verts.Count][];
            for (int i = 0; i < verts.Count; i++)
                snapped[i] = new[] { snap(verts[i][0]), snap(verts[i][1]), snap(verts[i][2]) };

            Comparison<float[]> cmp = (a, b) =>
            {
                if (a[0] != b[0]) return a[0] < b[0] ? -1 : 1;
                if (a[1] != b[1]) return a[1] < b[1] ? -1 : 1;
                if (a[2] != b[2]) return a[2] < b[2] ? -1 : 1;
                return 0;
            };

            var sorted = new float[snapped.Length][];
            Array.Copy(snapped, sorted, snapped.Length);
            Array.Sort(sorted, cmp);
            // unique (dedup consecutive)
            uniq = new List<float[]>(sorted.Length);
            for (int i = 0; i < sorted.Length; i++)
                if (i == 0 || cmp(sorted[i], sorted[i - 1]) != 0) uniq.Add(sorted[i]);

            // lower_bound of each original snapped vertex within uniq
            triIndex = new int[snapped.Length];
            var u = uniq;
            for (int i = 0; i < snapped.Length; i++)
            {
                int lo = 0, hi = u.Count;
                var v = snapped[i];
                while (lo < hi)
                {
                    int mid = (lo + hi) / 2;
                    if (cmp(u[mid], v) < 0) lo = mid + 1; else hi = mid;
                }
                triIndex[i] = lo;
            }
        }

        // OBJ winding: CUDA emits "f y+1 x+1 z+1"
        static void WriteObj(string path, List<float[]> verts, List<int[]> faces)
        {
            var sb = new StringBuilder(verts.Count * 24 + faces.Count * 24);
            foreach (var v in verts)
                sb.Append("v ").Append(v[0].ToString("F3", CultureInfo.InvariantCulture)).Append(' ')
                  .Append(v[1].ToString("F3", CultureInfo.InvariantCulture)).Append(' ')
                  .Append(v[2].ToString("F3", CultureInfo.InvariantCulture)).Append('\n');
            sb.Append('\n');
            foreach (var f in faces)
                sb.Append("f ").Append(f[1] + 1).Append(' ').Append(f[0] + 1).Append(' ').Append(f[2] + 1).Append('\n');
            File.WriteAllText(path, sb.ToString());
        }
    }
}
