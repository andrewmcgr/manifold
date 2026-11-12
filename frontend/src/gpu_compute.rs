use bytemuck::{Pod, Zeroable};
use std::sync::{Mutex, OnceLock};
use wgpu::util::DeviceExt;

// ============================================================================
// WGSL Compute Shader 1: Wavefront Constrained Eikonal & Clearance Cone
// ============================================================================

const EIKONAL_WAVEFRONT_SHADER_WGSL: &str = r#"
struct WavefrontUniforms {
    begin_offset: u32,
    count: u32,
    use_directional: u32,
    dx: f32,
};

@group(0) @binding(0) var<uniform> params: WavefrontUniforms;
@group(0) @binding(1) var<storage, read> wf_indices: array<u32>;
@group(0) @binding(2) var<storage, read> neighbors6: array<i32>;
@group(0) @binding(3) var<storage, read> voxel_flags: array<u32>;
@group(0) @binding(4) var<storage, read> slowness_target_dir: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> tau: array<f32>;

const K_INFINITY_TAU: f32 = 1.0e6;

fn neighbor_tau_isotropic(idx: u32, n_idx: i32, idx_is_interior: bool) -> f32 {
    if (n_idx < 0) {
        return K_INFINITY_TAU;
    }
    let u = u32(n_idx);
    let u_flags = voxel_flags[u];
    let u_is_interior = (u_flags & 1u) != 0u;
    if (idx_is_interior && !u_is_interior) {
        return K_INFINITY_TAU;
    }
    return tau[u];
}

fn godunov_isotropic_update(idx: u32, flags: u32) -> f32 {
    if ((flags & 2u) != 0u) {
        return tau[idx];
    }
    let idx_is_interior = (flags & 1u) != 0u;
    let s_dir = slowness_target_dir[idx];
    let step_cost = max(0.15, s_dir.w) * params.dx;
    let base_n = idx * 6u;

    let ax = min(
        neighbor_tau_isotropic(idx, neighbors6[base_n + 0u], idx_is_interior),
        neighbor_tau_isotropic(idx, neighbors6[base_n + 1u], idx_is_interior)
    );
    let ay = min(
        neighbor_tau_isotropic(idx, neighbors6[base_n + 2u], idx_is_interior),
        neighbor_tau_isotropic(idx, neighbors6[base_n + 3u], idx_is_interior)
    );
    let az = min(
        neighbor_tau_isotropic(idx, neighbors6[base_n + 4u], idx_is_interior),
        neighbor_tau_isotropic(idx, neighbors6[base_n + 5u], idx_is_interior)
    );

    // Sort 3 elements (a0 <= a1 <= a2)
    var a0 = ax;
    var a1 = ay;
    var a2 = az;
    if (a0 > a1) { let t = a0; a0 = a1; a1 = t; }
    if (a1 > a2) { let t = a1; a1 = a2; a2 = t; }
    if (a0 > a1) { let t = a0; a0 = a1; a1 = t; }

    if (a0 >= 0.5 * K_INFINITY_TAU) {
        return tau[idx];
    }

    var u = a0 + step_cost;
    if (u > a1) {
        let diff12 = a0 - a1;
        let disc2 = 2.0 * step_cost * step_cost - diff12 * diff12;
        if (disc2 >= 0.0) {
            u = 0.5 * (a0 + a1 + sqrt(disc2));
        }
        if (u > a2) {
            let sum3 = a0 + a1 + a2;
            let sq_sum3 = a0 * a0 + a1 * a1 + a2 * a2;
            let disc3 = sum3 * sum3 - 3.0 * (sq_sum3 - step_cost * step_cost);
            if (disc3 >= 0.0) {
                u = (sum3 + sqrt(disc3)) / 3.0;
            }
        }
    }
    return u;
}

fn directional_gradient_update(idx: u32) -> f32 {
    let dx = params.dx;
    let s_dir_idx = slowness_target_dir[idx];
    let s_idx = max(0.15, s_dir_idx.w);
    let d_idx = s_dir_idx.xyz;
    let base_n = idx * 6u;

    var step_from_u = array<vec3<f32>, 6>(
        vec3<f32>(dx, 0.0, 0.0),
        vec3<f32>(-dx, 0.0, 0.0),
        vec3<f32>(0.0, dx, 0.0),
        vec3<f32>(0.0, -dx, 0.0),
        vec3<f32>(0.0, 0.0, dx),
        vec3<f32>(0.0, 0.0, -dx)
    );

    var weighted_tau_sum = 0.0;
    var weight_sum = 0.0;

    for (var k = 0u; k < 6u; k = k + 1u) {
        let nb = neighbors6[base_n + k];
        if (nb < 0) {
            continue;
        }
        let u = u32(nb);
        let u_flags = voxel_flags[u];
        let u_tau = tau[u];
        if ((u_flags & 1u) == 0u || u_tau >= 0.5 * K_INFINITY_TAU) {
            continue;
        }
        let s_dir_u = slowness_target_dir[u];
        let s_u = max(0.15, s_dir_u.w);
        let d_u = s_dir_u.xyz;
        let g_edge = 0.5 * (s_idx * d_idx + s_u * d_u);
        let delta_tau = dot(g_edge, step_from_u[k]);
        let w = select(0.15, 1.0 + 5.0 * (delta_tau / dx), delta_tau >= 0.0);
        weighted_tau_sum = weighted_tau_sum + w * (u_tau + delta_tau);
        weight_sum = weight_sum + w;
    }

    if (weight_sum > 1.0e-5) {
        return weighted_tau_sum / weight_sum;
    }
    return tau[idx];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count) {
        return;
    }
    let idx = wf_indices[params.begin_offset + gid.x];
    let flags = voxel_flags[idx];
    // Bit 2 is isPinnedSeed
    if ((flags & 4u) != 0u) {
        return;
    }
    if (params.use_directional == 0u) {
        let cand = godunov_isotropic_update(idx, flags);
        if (cand < tau[idx]) {
            tau[idx] = cand;
        }
    } else {
        tau[idx] = directional_gradient_update(idx);
    }
}
"#;

// ============================================================================
// WGSL Compute Shader 2: 3D Tool-Clearance Collision Sweep (NozzleFrame)
// ============================================================================

const TOOL_CLEARANCE_SWEEP_SHADER_WGSL: &str = r#"
struct ClearanceUniforms {
    candidate_count: u32,
    clearance_mode: u32,
    profile_count: u32,
    brick_nx: i32,
    brick_ny: i32,
    brick_nz: i32,
    dense_nx: i32,
    dense_ny: i32,
    dense_nz: i32,
    use_dense_grid: u32,
    pad0: u32,
    pad1: u32,

    nozzle_axis_bed: vec4<f32>,
    travel_axis_bed: vec4<f32>,
    lateral_axis_bed: vec4<f32>,

    solid_bbox_min: vec4<f32>,
    solid_bbox_max: vec4<f32>,
    brick_origin_inv: vec4<f32>,
    dense_origin_inv: vec4<f32>,

    // scalars0: (tip_radius_mm, wall_half_width_mm, fine_voxel_size, h_top)
    scalars0: vec4<f32>,
    // scalars1: (max_tool_clearance_height_mm, max_plausible_tau, voxel_size, radial_step)
    scalars1: vec4<f32>,
    // cone_rect_params0: (cone_tip_radius_mm, cone_slope, rect_tip_hx, rect_tip_hy)
    cone_rect_params0: vec4<f32>,
    // cone_rect_params1: (rect_slope_x, rect_slope_y, profile_end_slope, 0.0)
    cone_rect_params1: vec4<f32>,

    profile_xz: array<vec4<f32>, 8>,
};

@group(0) @binding(0) var<uniform> u: ClearanceUniforms;
// candidate_pos_depth[i] = (xSurf.x, xSurf.y, xSurf.z, skinDepthMm)
@group(0) @binding(1) var<storage, read> candidate_pos_depth: array<vec4<f32>>;
// candidate_norm_tau[i] = (nSdf.x, nSdf.y, nSdf.z, printTau)
@group(0) @binding(2) var<storage, read> candidate_norm_tau: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> brick_min_tau: array<f32>;
// dense_sdf_tau[idx] = vec2<f32>(sdf, tau) at cell-centered fine grid
@group(0) @binding(4) var<storage, read> dense_sdf_tau: array<vec2<f32>>;
// out_flags[i]: bit 0 = collides, bit 1 = near_boundary_margin
@group(0) @binding(5) var<storage, read_write> out_flags: array<u32>;

fn eval_tool_clearance_radius_at_z(z_local_mm: f32) -> f32 {
    let z = max(0.0, z_local_mm);
    let tip_radius = u.scalars0.x;
    let wall_half_w = u.scalars0.y;
    let fine_voxel = u.scalars0.z;
    var r_tool = tip_radius;

    if (u.clearance_mode == 1u) {
        r_tool = max(r_tool, u.cone_rect_params0.x + z * u.cone_rect_params0.y);
    } else if (u.clearance_mode == 2u) {
        let hx = u.cone_rect_params0.z + z * u.cone_rect_params1.x;
        let hy = u.cone_rect_params0.w + z * u.cone_rect_params1.y;
        r_tool = max(r_tool, sqrt(hx * hx + hy * hy));
    } else if (u.profile_count > 0u) {
        let n = min(u.profile_count, 16u);
        let p0_pair = u.profile_xz[0u];
        let r_first = p0_pair.x;
        let z_first = p0_pair.y;
        let last_idx = n - 1u;
        let last_vec = u.profile_xz[last_idx / 2u];
        let r_last = select(last_vec.z, last_vec.x, (last_idx & 1u) == 0u);
        let z_last = select(last_vec.w, last_vec.y, (last_idx & 1u) == 0u);

        if (z <= z_first) {
            r_tool = max(r_tool, r_first);
        } else if (z >= z_last) {
            r_tool = max(r_tool, r_last + (z - z_last) * u.cone_rect_params1.z);
        } else {
            for (var k = 0u; k + 1u < n; k = k + 1u) {
                let v0 = u.profile_xz[k / 2u];
                let r0 = select(v0.z, v0.x, (k & 1u) == 0u);
                let z0 = select(v0.w, v0.y, (k & 1u) == 0u);
                let k1 = k + 1u;
                let v1 = u.profile_xz[k1 / 2u];
                let r1 = select(v1.z, v1.x, (k1 & 1u) == 0u);
                let z1 = select(v1.w, v1.y, (k1 & 1u) == 0u);
                if (z >= z0 && z <= z1) {
                    let dz = z1 - z0;
                    let t = select(0.0, (z - z0) / dz, dz > 1.0e-5);
                    r_tool = max(r_tool, r0 + t * (r1 - r0));
                    break;
                }
            }
        }
    } else {
        r_tool = max(r_tool, tip_radius + 0.65 * z);
    }

    return r_tool + wall_half_w + 0.35 * fine_voxel;
}

fn is_brick_possibly_occupied(pt: vec3<f32>, max_allowed_tau: f32) -> bool {
    if (pt.x < u.solid_bbox_min.x || pt.x > u.solid_bbox_max.x ||
        pt.y < u.solid_bbox_min.y || pt.y > u.solid_bbox_max.y ||
        pt.z < u.solid_bbox_min.z || pt.z > u.solid_bbox_max.z) {
        return false;
    }
    let inv_b = u.brick_origin_inv.w;
    let bx = i32(floor((pt.x - u.brick_origin_inv.x) * inv_b));
    let by = i32(floor((pt.y - u.brick_origin_inv.y) * inv_b));
    let bz = i32(floor((pt.z - u.brick_origin_inv.z) * inv_b));
    if (bx < 0 || bx >= u.brick_nx ||
        by < 0 || by >= u.brick_ny ||
        bz < 0 || bz >= u.brick_nz) {
        return false;
    }
    let stride_y = u32(u.brick_nx);
    let stride_z = stride_y * u32(u.brick_ny);
    let b_idx = u32(bz) * stride_z + u32(by) * stride_y + u32(bx);
    return brick_min_tau[b_idx] <= max_allowed_tau;
}

fn fetch_dense_cell(ix: i32, iy: i32, iz: i32) -> vec2<f32> {
    let cx = clamp(ix, 0, u.dense_nx - 1);
    let cy = clamp(iy, 0, u.dense_ny - 1);
    let cz = clamp(iz, 0, u.dense_nz - 1);
    let stride_y = u32(u.dense_nx);
    let stride_z = stride_y * u32(u.dense_ny);
    let idx = u32(cz) * stride_z + u32(cy) * stride_y + u32(cx);
    return dense_sdf_tau[idx];
}

fn sample_dense_sdf_and_tau(pt: vec3<f32>) -> vec2<f32> {
    let inv_h = u.dense_origin_inv.w;
    let gx = (pt.x - u.dense_origin_inv.x) * inv_h;
    let gy = (pt.y - u.dense_origin_inv.y) * inv_h;
    let gz = (pt.z - u.dense_origin_inv.z) * inv_h;

    let i0 = i32(floor(gx));
    let j0 = i32(floor(gy));
    let k0 = i32(floor(gz));

    if (i0 < 0 || i0 >= u.dense_nx - 1 ||
        j0 < 0 || j0 >= u.dense_ny - 1 ||
        k0 < 0 || k0 >= u.dense_nz - 1) {
        return fetch_dense_cell(i32(round(gx)), i32(round(gy)), i32(round(gz)));
    }

    let fx = gx - f32(i0);
    let fy = gy - f32(j0);
    let fz = gz - f32(k0);

    let c000 = fetch_dense_cell(i0, j0, k0);
    let c100 = fetch_dense_cell(i0 + 1, j0, k0);
    let c010 = fetch_dense_cell(i0, j0 + 1, k0);
    let c110 = fetch_dense_cell(i0 + 1, j0 + 1, k0);
    let c001 = fetch_dense_cell(i0, j0, k0 + 1);
    let c101 = fetch_dense_cell(i0 + 1, j0, k0 + 1);
    let c011 = fetch_dense_cell(i0, j0 + 1, k0 + 1);
    let c111 = fetch_dense_cell(i0 + 1, j0 + 1, k0 + 1);

    let c00 = mix(c000, c100, fx);
    let c10 = mix(c010, c110, fx);
    let c01 = mix(c001, c101, fx);
    let c11 = mix(c011, c111, fx);
    let c0 = mix(c00, c10, fy);
    let c1 = mix(c01, c11, fy);
    let interp = mix(c0, c1, fz);

    let max_plausible_tau = u.scalars1.y;
    var tau_val = interp.y;
    if (tau_val < 0.0 || tau_val > max_plausible_tau) {
        // Neighborhood fallback when a trilinear corner touches background
        let ci = i32(round(gx));
        let cj = i32(round(gy));
        let ck = i32(round(gz));
        var best_tau = 1.0e9;
        var best_d2 = 100;
        for (var dz = -1; dz <= 1; dz = dz + 1) {
            for (var dy = -1; dy <= 1; dy = dy + 1) {
                for (var dx = -1; dx <= 1; dx = dx + 1) {
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if (d2 > best_d2) {
                        continue;
                    }
                    let cell = fetch_dense_cell(ci + dx, cj + dy, ck + dz);
                    if (cell.y >= 0.0 && cell.y <= max_plausible_tau) {
                        if (d2 < best_d2 || cell.y < best_tau) {
                            best_d2 = d2;
                            best_tau = cell.y;
                        }
                    }
                }
            }
        }
        tau_val = select(0.0, best_tau, best_tau <= max_plausible_tau);
    }

    return vec2<f32>(interp.x, tau_val);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.candidate_count) {
        return;
    }

    let pos_depth = candidate_pos_depth[idx];
    let norm_tau = candidate_norm_tau[idx];
    let x_surf = pos_depth.xyz;
    let skin_depth_mm = pos_depth.w;
    let n_sdf = norm_tau.xyz;
    let print_tau = norm_tau.w;

    let z_axis = u.nozzle_axis_bed.xyz;
    let x_axis = u.travel_axis_bed.xyz;
    let y_axis = u.lateral_axis_bed.xyz;

    let x_base = x_surf - skin_depth_mm * n_sdf;

    let bmin = u.solid_bbox_min.xyz;
    let bmax = u.solid_bbox_max.xyz;
    var max_z_along_nozzle = 0.0;
    for (var c = 0u; c < 8u; c = c + 1u) {
        let corner = vec3<f32>(
            select(bmin.x, bmax.x, (c & 1u) != 0u),
            select(bmin.y, bmax.y, (c & 2u) != 0u),
            select(bmin.z, bmax.z, (c & 4u) != 0u)
        );
        max_z_along_nozzle = max(max_z_along_nozzle, dot(corner - x_base, z_axis));
    }

    let fine_voxel = u.scalars0.z;
    let h_top = u.scalars0.w;
    let max_tool_h = u.scalars1.x;
    let max_plausible_tau = u.scalars1.y;
    let radial_step = u.scalars1.w;

    let z_sweep_max = min(max_z_along_nozzle, min(max_tool_h, 12.0));
    let z_start = max(0.60 * h_top, 0.50 * fine_voxel);
    if (z_sweep_max <= z_start) {
        out_flags[idx] = 0u;
        return;
    }

    let near_tip_limit = skin_depth_mm + max(1.75 * h_top, 1.35 * fine_voxel);
    let print_tau_tol = print_tau + 0.65 * h_top;
    let base_curv_tol = max(0.45 * h_top, 0.35 * fine_voxel);
    let margin_extra = max(0.65 * fine_voxel, 0.85 * h_top);

    var dirs12 = array<vec2<f32>, 12>(
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.8660254, 0.5),
        vec2<f32>(0.5, 0.8660254),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(-0.5, 0.8660254),
        vec2<f32>(-0.8660254, 0.5),
        vec2<f32>(-1.0, 0.0),
        vec2<f32>(-0.8660254, -0.5),
        vec2<f32>(-0.5, -0.8660254),
        vec2<f32>(0.0, -1.0),
        vec2<f32>(0.5, -0.8660254),
        vec2<f32>(0.8660254, -0.5)
    );

    var near_margin = false;
    var z = z_start;
    loop {
        if (z > z_sweep_max) {
            break;
        }
        let r_clear = eval_tool_clearance_radius_at_z(z);
        let r_margin = select(r_clear, r_clear + margin_extra, z <= near_tip_limit + 2.0 * h_top);
        let p_axis = x_base + z * z_axis;
        let slice_tau_limit = select(print_tau_tol, max_plausible_tau, z <= near_tip_limit);
        let z_step = select(
            max(0.95 * fine_voxel, 0.65),
            max(0.55 * fine_voxel, 0.65 * h_top),
            z < near_tip_limit + 2.0 * h_top
        );

        if (p_axis.x + r_margin < bmin.x || p_axis.x - r_margin > bmax.x ||
            p_axis.y + r_margin < bmin.y || p_axis.y - r_margin > bmax.y ||
            p_axis.z + r_margin < bmin.z || p_axis.z - r_margin > bmax.z) {
            z = z + z_step;
            continue;
        }

        if (z > skin_depth_mm + 0.55 * h_top) {
            let h_above_surf_axis = dot(p_axis - x_surf, n_sdf);
            if (h_above_surf_axis > base_curv_tol &&
                is_brick_possibly_occupied(p_axis, slice_tau_limit)) {
                let st = sample_dense_sdf_and_tau(p_axis);
                if (st.x <= 0.08 * fine_voxel) {
                    if (z <= near_tip_limit || st.y <= print_tau_tol) {
                        out_flags[idx] = 1u;
                        return;
                    }
                }
            }
        }

        let num_rings = max(1, i32(ceil(r_margin / radial_step)));
        for (var ri = 1; ri <= num_rings; ri = ri + 1) {
            let r = (f32(ri) / f32(num_rings)) * r_margin;
            let is_margin_ring = r > r_clear + 1.0e-4;
            if (is_margin_ring && near_margin) {
                continue;
            }
            let curv_tol = base_curv_tol + 0.03 * r;
            let check_steep_core = z > skin_depth_mm + 0.75 * h_top;

            for (var d = 0u; d < 12u; d = d + 1u) {
                let dir = dirs12[d];
                let r_vec = x_axis * dir.x + y_axis * dir.y;
                let y_pos = p_axis + r * r_vec;

                let h_above_surf = dot(y_pos - x_surf, n_sdf);
                let h_above_base = h_above_surf + skin_depth_mm;
                let above_surf = h_above_surf > curv_tol;
                let above_base = check_steep_core && (h_above_base > curv_tol);
                if (!above_surf && !above_base) {
                    continue;
                }

                let max_allowed_tau = select(
                    print_tau_tol,
                    max_plausible_tau,
                    above_surf && (z <= near_tip_limit)
                );
                if (!is_brick_possibly_occupied(y_pos, max_allowed_tau)) {
                    continue;
                }

                let st = sample_dense_sdf_and_tau(y_pos);
                if (st.x > 0.08 * fine_voxel) {
                    continue;
                }

                var hits_obstacle = false;
                if (above_surf) {
                    if (z <= near_tip_limit || st.y <= print_tau_tol) {
                        hits_obstacle = true;
                    }
                } else if (above_base && st.x <= -(skin_depth_mm + 0.35 * h_top)) {
                    if (st.y <= print_tau_tol) {
                        hits_obstacle = true;
                    }
                }

                if (hits_obstacle) {
                    if (!is_margin_ring) {
                        out_flags[idx] = 1u;
                        return;
                    }
                    near_margin = true;
                }
            }
        }

        z = z + z_step;
    }

    out_flags[idx] = select(0u, 2u, near_margin);
}
"#;

// ============================================================================
// WGSL Compute Shader 3: Batch 3D TPMS Evaluation (Gyroid, SchwarzD, SchwarzP)
// ============================================================================

const BATCH_TPMS_SHADER_WGSL: &str = r#"
struct TpmsUniforms {
    vertex_count: u32,
    pattern_kind: u32,
    wave_k: f32,
    pad0: f32,
};

@group(0) @binding(0) var<uniform> params: TpmsUniforms;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_psi: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.vertex_count) {
        return;
    }
    let base = i * 3u;
    let kx = params.wave_k * positions[base + 0u];
    let ky = params.wave_k * positions[base + 1u];
    let kz = params.wave_k * positions[base + 2u];

    var psi = 0.0;
    if (params.pattern_kind == 2u) {
        // Gyroid
        psi = sin(kx) * cos(ky) + sin(ky) * cos(kz) + sin(kz) * cos(kx);
    } else if (params.pattern_kind == 3u) {
        // Schwarz D
        psi = sin(kx) * sin(ky) * sin(kz) +
              sin(kx) * cos(ky) * cos(kz) +
              cos(kx) * sin(ky) * cos(kz) +
              cos(kx) * cos(ky) * sin(kz);
    } else {
        // Schwarz P
        psi = cos(kx) + cos(ky) + cos(kz);
    }
    out_psi[i] = psi;
}
"#;

// ============================================================================
// WGSL Compute Shader 4: 5/6-Axis Tool Axis Jacobi Smoothing & Cone Clamping
// ============================================================================

const TOOL_AXIS_SMOOTH_SHADER_WGSL: &str = r#"
struct SmoothAxisUniforms {
    voxel_count: u32,
    compute_tilt_deg: u32,
    pad0: u32,
    pad1: u32,
    bed_normal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: SmoothAxisUniforms;
@group(0) @binding(1) var<storage, read> neighbors6: array<i32>;
@group(0) @binding(2) var<storage, read> max_tilt_rad: array<f32>;
@group(0) @binding(3) var<storage, read> in_axes: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> out_axes: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> out_tilt_deg: array<f32>;

const K_PI: f32 = 3.14159265358979323846;
const K_RAD_TO_DEG: f32 = 57.29577951308232;

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if (len > 1.0e-7) {
        return v / len;
    }
    return fallback;
}

fn clamp_unit_vector_to_cone(u: vec3<f32>, ref_axis: vec3<f32>, max_angle_rad: f32) -> vec3<f32> {
    let clamped_max = clamp(max_angle_rad, 0.0, 0.495 * K_PI);
    if (clamped_max <= 1.0e-6) {
        return ref_axis;
    }
    let cos_theta = clamp(dot(u, ref_axis), -1.0, 1.0);
    let cos_max = cos(clamped_max);
    if (cos_theta >= cos_max) {
        return u;
    }
    let perp_raw = u - cos_theta * ref_axis;
    let perp_len = length(perp_raw);
    if (perp_len <= 1.0e-6) {
        return ref_axis;
    }
    let sin_max = sin(clamped_max);
    let clamped = cos_max * ref_axis + (sin_max / perp_len) * perp_raw;
    return safe_normalize(clamped, ref_axis);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.voxel_count) {
        return;
    }
    let bed_normal = params.bed_normal.xyz;
    let base_n = i * 6u;
    var acc = 2.0 * in_axes[i].xyz;
    for (var k = 0u; k < 6u; k = k + 1u) {
        let nb = neighbors6[base_n + k];
        if (nb >= 0) {
            acc = acc + in_axes[u32(nb)].xyz;
        }
    }
    let unit_avg = safe_normalize(acc, bed_normal);
    let clamped = clamp_unit_vector_to_cone(unit_avg, bed_normal, max_tilt_rad[i]);
    out_axes[i] = vec4<f32>(clamped, 0.0);
    if (params.compute_tilt_deg != 0u) {
        let cos_tilt = clamp(dot(clamped, bed_normal), -1.0, 1.0);
        out_tilt_deg[i] = acos(cos_tilt) * K_RAD_TO_DEG;
    }
}
"#;

// ============================================================================
// Uniform & Storage Pod Structs
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct WavefrontUniforms {
    begin_offset: u32,
    count: u32,
    use_directional: u32,
    dx: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ClearanceUniforms {
    pub candidate_count: u32,
    pub clearance_mode: u32,
    pub profile_count: u32,
    pub brick_nx: i32,
    pub brick_ny: i32,
    pub brick_nz: i32,
    pub dense_nx: i32,
    pub dense_ny: i32,
    pub dense_nz: i32,
    pub use_dense_grid: u32,
    pub pad0: u32,
    pub pad1: u32,

    pub nozzle_axis_bed: [f32; 4],
    pub travel_axis_bed: [f32; 4],
    pub lateral_axis_bed: [f32; 4],

    pub solid_bbox_min: [f32; 4],
    pub solid_bbox_max: [f32; 4],
    pub brick_origin_inv: [f32; 4],
    pub dense_origin_inv: [f32; 4],

    pub scalars0: [f32; 4],
    pub scalars1: [f32; 4],
    pub cone_rect_params0: [f32; 4],
    pub cone_rect_params1: [f32; 4],

    pub profile_xz: [[f32; 4]; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TpmsUniforms {
    vertex_count: u32,
    pattern_kind: u32,
    wave_k: f32,
    pad0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SmoothAxisUniforms {
    voxel_count: u32,
    compute_tilt_deg: u32,
    pad0: u32,
    pad1: u32,
    bed_normal: [f32; 4],
}

/// Thread-safe cross-platform `wgpu` compute engine with pre-compiled WGSL compute pipelines.
pub struct GpuComputeEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    eikonal_bgl: wgpu::BindGroupLayout,
    eikonal_pipeline: wgpu::ComputePipeline,
    clearance_bgl: wgpu::BindGroupLayout,
    clearance_pipeline: wgpu::ComputePipeline,
    tpms_bgl: wgpu::BindGroupLayout,
    tpms_pipeline: wgpu::ComputePipeline,
    axis_smooth_bgl: wgpu::BindGroupLayout,
    axis_smooth_pipeline: wgpu::ComputePipeline,
    lock: Mutex<()>,
}

static GPU_ENGINE: OnceLock<Option<GpuComputeEngine>> = OnceLock::new();

impl GpuComputeEngine {
    /// Lazily initializes the global headless-capable `wgpu` compute engine, or returns `None`
    /// if no hardware GPU adapter is available (e.g., inside restricted sandboxed environments).
    pub fn global() -> Option<&'static GpuComputeEngine> {
        GPU_ENGINE
            .get_or_init(|| {
                if std::env::var("MANIFOLD_DISABLE_GPU").is_ok() {
                    return None;
                }
                std::panic::catch_unwind(Self::try_init).ok().flatten()
            })
            .as_ref()
    }

    fn try_init() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = bevy::tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let limits = adapter.limits();
        if limits.max_storage_buffers_per_shader_stage < 5 {
            return None;
        }
        let (device, queue) = bevy::tasks::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("manifold_gpu_compute_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: limits
                        .max_storage_buffers_per_shader_stage
                        .max(8),
                    max_storage_buffer_binding_size: limits
                        .max_storage_buffer_binding_size
                        .min(512 * 1024 * 1024),
                    max_buffer_size: limits.max_buffer_size.min(512 * 1024 * 1024),
                    max_uniform_buffer_binding_size: limits.max_uniform_buffer_binding_size,
                    ..wgpu::Limits::downlevel_defaults()
                },
            },
            None,
        ))
        .ok()?;

        // 1. Eikonal Wavefront Pipeline
        let eikonal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("manifold_eikonal_wavefront_shader"),
            source: wgpu::ShaderSource::Wgsl(EIKONAL_WAVEFRONT_SHADER_WGSL.into()),
        });

        let eikonal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("manifold_eikonal_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<WavefrontUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
            ],
        });

        let eikonal_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("manifold_eikonal_pipeline_layout"),
            bind_group_layouts: &[&eikonal_bgl],
            push_constant_ranges: &[],
        });

        let eikonal_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("manifold_eikonal_compute_pipeline"),
            layout: Some(&eikonal_layout),
            module: &eikonal_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // 2. 3D Tool-Clearance Collision Sweep Pipeline
        let clearance_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("manifold_tool_clearance_sweep_shader"),
            source: wgpu::ShaderSource::Wgsl(TOOL_CLEARANCE_SWEEP_SHADER_WGSL.into()),
        });

        let clearance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("manifold_clearance_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<ClearanceUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
            ],
        });

        let clearance_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("manifold_clearance_pipeline_layout"),
            bind_group_layouts: &[&clearance_bgl],
            push_constant_ranges: &[],
        });

        let clearance_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("manifold_clearance_compute_pipeline"),
            layout: Some(&clearance_layout),
            module: &clearance_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // 3. Batch TPMS Pipeline
        let tpms_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("manifold_batch_tpms_shader"),
            source: wgpu::ShaderSource::Wgsl(BATCH_TPMS_SHADER_WGSL.into()),
        });

        let tpms_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("manifold_tpms_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<TpmsUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });

        let tpms_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("manifold_tpms_pipeline_layout"),
            bind_group_layouts: &[&tpms_bgl],
            push_constant_ranges: &[],
        });

        let tpms_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("manifold_tpms_compute_pipeline"),
            layout: Some(&tpms_layout),
            module: &tpms_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // 4. 5/6-Axis Tool Axis Jacobi Smoothing & Cone Clamping Pipeline
        let axis_smooth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("manifold_tool_axis_smooth_shader"),
            source: wgpu::ShaderSource::Wgsl(TOOL_AXIS_SMOOTH_SHADER_WGSL.into()),
        });

        let axis_smooth_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("manifold_axis_smooth_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<SmoothAxisUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
        });

        let axis_smooth_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("manifold_axis_smooth_pipeline_layout"),
            bind_group_layouts: &[&axis_smooth_bgl],
            push_constant_ranges: &[],
        });

        let axis_smooth_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("manifold_axis_smooth_compute_pipeline"),
                layout: Some(&axis_smooth_layout),
                module: &axis_smooth_shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        Some(Self {
            device,
            queue,
            eikonal_bgl,
            eikonal_pipeline,
            clearance_bgl,
            clearance_pipeline,
            tpms_bgl,
            tpms_pipeline,
            axis_smooth_bgl,
            axis_smooth_pipeline,
            lock: Mutex::new(()),
        })
    }

    /// Executes 8-octant hyperplane wavefront Eikonal sweeps on the GPU over `tau_io`.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_eikonal_wavefronts(
        &self,
        neighbors6: &[i32],
        voxel_flags: &[u32],
        slowness: &[f32],
        target_dir: &[f32],
        wf_indices: &[u32],
        wf_level_offsets: &[u32],
        wf_octant_bounds: &[u32],
        dx: f32,
        num_sweep_rounds: u32,
        use_directional_guidance: bool,
        tau_io: &mut [f32],
    ) -> bool {
        let n = tau_io.len();
        let max_storage = self.device.limits().max_storage_buffer_binding_size as usize;
        let max_wg = self
            .device
            .limits()
            .max_compute_workgroups_per_dimension
            .min(65_535);
        if n < 2048
            || n.saturating_mul(24) > max_storage
            || neighbors6.len() != n * 6
            || voxel_flags.len() != n
            || slowness.len() != n
            || target_dir.len() != n * 3
            || wf_octant_bounds.len() < 5
            || wf_indices.is_empty()
        {
            return false;
        }

        let _guard = match self.lock.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };

        let mut slowness_target = Vec::with_capacity(n * 4);
        for i in 0..n {
            slowness_target.push(target_dir[i * 3]);
            slowness_target.push(target_dir[i * 3 + 1]);
            slowness_target.push(target_dir[i * 3 + 2]);
            slowness_target.push(slowness[i]);
        }

        // Build aligned dynamic uniform buffer for all wavefront slices in one octant round
        let align = self
            .device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(256) as usize;

        let mut dispatches: Vec<(u32, u32)> = Vec::new();
        for q in 0..4usize {
            let l_start = wf_octant_bounds[q] as usize;
            let l_end = wf_octant_bounds[q + 1] as usize;
            if l_end <= l_start + 1 || l_end > wf_level_offsets.len() {
                continue;
            }
            let num_lvls = l_end - l_start - 1;
            // Forward octant sweep
            for lvl in 0..num_lvls {
                let b_off = wf_level_offsets[l_start + lvl];
                let e_off = wf_level_offsets[l_start + lvl + 1];
                if e_off > b_off {
                    dispatches.push((b_off, e_off - b_off));
                }
            }
            // Reverse octant sweep
            for lvl in (0..num_lvls).rev() {
                let b_off = wf_level_offsets[l_start + lvl];
                let e_off = wf_level_offsets[l_start + lvl + 1];
                if e_off > b_off {
                    dispatches.push((b_off, e_off - b_off));
                }
            }
        }

        if dispatches.is_empty()
            || dispatches
                .iter()
                .any(|&(_, count)| count.div_ceil(64) > max_wg)
        {
            return false;
        }

        let max_buf_size = self.device.limits().max_buffer_size as usize;
        let total_uniform_bytes = dispatches.len() * align;
        if total_uniform_bytes > max_buf_size && max_buf_size > 0 {
            return false;
        }

        let mut uniform_bytes = vec![0u8; total_uniform_bytes];
        let use_dir_u32 = u32::from(use_directional_guidance);
        for (idx, &(begin_offset, count)) in dispatches.iter().enumerate() {
            let u = WavefrontUniforms {
                begin_offset,
                count,
                use_directional: use_dir_u32,
                dx,
            };
            let dst = &mut uniform_bytes[idx * align..idx * align + std::mem::size_of::<WavefrontUniforms>()];
            dst.copy_from_slice(bytemuck::bytes_of(&u));
        }

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("eikonal_wf_uniform_buf"),
                contents: &uniform_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let wf_ind_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("eikonal_wf_indices_buf"),
                contents: bytemuck::cast_slice(wf_indices),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let nb6_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("eikonal_nb6_buf"),
                contents: bytemuck::cast_slice(neighbors6),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let flags_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("eikonal_flags_buf"),
                contents: bytemuck::cast_slice(voxel_flags),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let s_dir_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("eikonal_s_dir_buf"),
                contents: bytemuck::cast_slice(&slowness_target),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let tau_byte_len = std::mem::size_of_val(tau_io) as u64;
        let tau_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("eikonal_tau_buf"),
                contents: bytemuck::cast_slice(tau_io),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("eikonal_tau_readback_buf"),
            size: tau_byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eikonal_wf_bind_group"),
            layout: &self.eikonal_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(
                            std::mem::size_of::<WavefrontUniforms>() as u64,
                        ),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wf_ind_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: nb6_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: flags_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: s_dir_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: tau_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("eikonal_wf_encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("eikonal_wf_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.eikonal_pipeline);

            for _round in 0..num_sweep_rounds.max(1) {
                for (d_idx, &(_, count)) in dispatches.iter().enumerate() {
                    let dyn_off = (d_idx * align) as u32;
                    cpass.set_bind_group(0, &bind_group, &[dyn_off]);
                    cpass.dispatch_workgroups(count.div_ceil(64), 1, 1);
                }
            }
        }

        encoder.copy_buffer_to_buffer(&tau_buf, 0, &readback_buf, 0, tau_byte_len);
        self.queue.submit(Some(encoder.finish()));

        read_back_buffer_f32(&self.device, &readback_buf, tau_io)
    }

    /// Evaluates 3D tool-clearance collision cones across `candidate_pos_depth` on the GPU.
    pub fn sweep_tool_clearance(
        &self,
        uniforms: &ClearanceUniforms,
        candidate_pos_depth: &[[f32; 4]],
        candidate_norm_tau: &[[f32; 4]],
        brick_min_tau: &[f32],
        dense_sdf_tau: &[[f32; 2]],
        out_flags: &mut [u32],
    ) -> bool {
        let n = candidate_pos_depth.len();
        let max_wg = self
            .device
            .limits()
            .max_compute_workgroups_per_dimension
            .min(65_535);
        if n == 0
            || (n as u32).div_ceil(64) > max_wg
            || candidate_norm_tau.len() != n
            || out_flags.len() != n
            || brick_min_tau.is_empty()
            || dense_sdf_tau.is_empty()
        {
            return false;
        }

        let max_storage = self.device.limits().max_storage_buffer_binding_size as usize;
        if std::mem::size_of_val(dense_sdf_tau) > max_storage
            || n.saturating_mul(16) > max_storage
        {
            return false;
        }

        let _guard = match self.lock.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("clearance_uniform_buf"),
                contents: bytemuck::bytes_of(uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let pos_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("clearance_pos_buf"),
                contents: bytemuck::cast_slice(candidate_pos_depth),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let norm_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("clearance_norm_buf"),
                contents: bytemuck::cast_slice(candidate_norm_tau),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let brick_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("clearance_brick_buf"),
                contents: bytemuck::cast_slice(brick_min_tau),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let dense_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("clearance_dense_buf"),
                contents: bytemuck::cast_slice(dense_sdf_tau),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let out_byte_len = std::mem::size_of_val(out_flags) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clearance_out_buf"),
            size: out_byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clearance_readback_buf"),
            size: out_byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clearance_bind_group"),
            layout: &self.clearance_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pos_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: norm_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: brick_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dense_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("clearance_encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clearance_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.clearance_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &readback_buf, 0, out_byte_len);
        self.queue.submit(Some(encoder.finish()));

        read_back_buffer_u32(&self.device, &readback_buf, out_flags)
    }

    /// Evaluates 3D TPMS implicit functions (`Gyroid = 2`, `SchwarzD = 3`, `SchwarzP = 4`)
    /// at a batch of 3D vertex positions (`positions_xyz`, length `3 * N`) on the GPU.
    pub fn evaluate_tpms_batch(
        &self,
        positions_xyz: &[f32],
        pattern_kind: u8,
        wave_k: f32,
        out_psi: &mut [f32],
    ) -> bool {
        let n = out_psi.len();
        let max_storage = self.device.limits().max_storage_buffer_binding_size as usize;
        let max_wg = self
            .device
            .limits()
            .max_compute_workgroups_per_dimension
            .min(65_535);
        if n < 512
            || (n as u32).div_ceil(64) > max_wg
            || n.saturating_mul(12) > max_storage
            || positions_xyz.len() != n * 3
        {
            return false;
        }

        let _guard = match self.lock.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };

        let u = TpmsUniforms {
            vertex_count: n as u32,
            pattern_kind: pattern_kind as u32,
            wave_k,
            pad0: 0.0,
        };

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tpms_uniform_buf"),
                contents: bytemuck::bytes_of(&u),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let pos_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tpms_positions_buf"),
                contents: bytemuck::cast_slice(positions_xyz),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_byte_len = std::mem::size_of_val(out_psi) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tpms_out_buf"),
            size: out_byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tpms_readback_buf"),
            size: out_byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tpms_bind_group"),
            layout: &self.tpms_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pos_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tpms_encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tpms_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.tpms_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &readback_buf, 0, out_byte_len);
        self.queue.submit(Some(encoder.finish()));

        read_back_buffer_f32(&self.device, &readback_buf, out_psi)
    }

    /// Runs `num_passes` Jacobi Laplacian smoothing + spherical cone-clamping passes
    /// over the 5/6-axis tool orientation field `axes_io` (`3 * N` floats) on the GPU,
    /// and writes the resulting tilt angles (degrees from `bed_normal_xyz`) to `out_tilt_deg`.
    pub fn smooth_and_clamp_tool_axes(
        &self,
        neighbors6: &[i32],
        bed_normal_xyz: &[f32],
        local_max_tilt_rad: &[f32],
        num_passes: u32,
        axes_io: &mut [f32],
        out_tilt_deg: &mut [f32],
    ) -> bool {
        let n = out_tilt_deg.len();
        let max_storage = self.device.limits().max_storage_buffer_binding_size as usize;
        let max_wg = self
            .device
            .limits()
            .max_compute_workgroups_per_dimension
            .min(65_535);
        if n < 1024
            || (n as u32).div_ceil(64) > max_wg
            || n.saturating_mul(24) > max_storage
            || neighbors6.len() != n * 6
            || bed_normal_xyz.len() < 3
            || local_max_tilt_rad.len() != n
            || axes_io.len() != n * 3
            || num_passes == 0
        {
            return false;
        }

        let _guard = match self.lock.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };

        let mut axes_vec4 = Vec::with_capacity(n * 4);
        for i in 0..n {
            axes_vec4.push(axes_io[i * 3]);
            axes_vec4.push(axes_io[i * 3 + 1]);
            axes_vec4.push(axes_io[i * 3 + 2]);
            axes_vec4.push(0.0f32);
        }

        let u_pass = SmoothAxisUniforms {
            voxel_count: n as u32,
            compute_tilt_deg: 1,
            pad0: 0,
            pad1: 0,
            bed_normal: [
                bed_normal_xyz[0],
                bed_normal_xyz[1],
                bed_normal_xyz[2],
                0.0,
            ],
        };

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("axis_smooth_uniform_buf"),
                contents: bytemuck::bytes_of(&u_pass),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let nb6_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("axis_smooth_nb6_buf"),
                contents: bytemuck::cast_slice(neighbors6),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let tilt_limit_buf =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("axis_smooth_max_tilt_buf"),
                    contents: bytemuck::cast_slice(local_max_tilt_rad),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let axes_byte_len = std::mem::size_of_val(axes_vec4.as_slice()) as u64;
        let axes_a_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("axis_smooth_a_buf"),
                contents: bytemuck::cast_slice(&axes_vec4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let axes_b_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("axis_smooth_b_buf"),
            size: axes_byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let tilt_byte_len = std::mem::size_of_val(out_tilt_deg) as u64;
        let tilt_out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("axis_smooth_tilt_out_buf"),
            size: tilt_byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let axes_readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("axis_smooth_axes_readback_buf"),
            size: axes_byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tilt_readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("axis_smooth_tilt_readback_buf"),
            size: tilt_byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg_ab = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("axis_smooth_bg_ab"),
            layout: &self.axis_smooth_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: nb6_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tilt_limit_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: axes_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: axes_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: tilt_out_buf.as_entire_binding(),
                },
            ],
        });

        let bg_ba = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("axis_smooth_bg_ba"),
            layout: &self.axis_smooth_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: nb6_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tilt_limit_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: axes_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: axes_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: tilt_out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("axis_smooth_encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("axis_smooth_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.axis_smooth_pipeline);
            let wg = (n as u32).div_ceil(64);
            for p in 0..num_passes {
                if p % 2 == 0 {
                    cpass.set_bind_group(0, &bg_ab, &[]);
                } else {
                    cpass.set_bind_group(0, &bg_ba, &[]);
                }
                cpass.dispatch_workgroups(wg, 1, 1);
            }
        }

        let final_axes_buf = if num_passes % 2 == 1 {
            &axes_b_buf
        } else {
            &axes_a_buf
        };
        encoder.copy_buffer_to_buffer(final_axes_buf, 0, &axes_readback_buf, 0, axes_byte_len);
        encoder.copy_buffer_to_buffer(&tilt_out_buf, 0, &tilt_readback_buf, 0, tilt_byte_len);
        self.queue.submit(Some(encoder.finish()));

        if !read_back_buffer_f32(&self.device, &axes_readback_buf, &mut axes_vec4) {
            return false;
        }
        if !read_back_buffer_f32(&self.device, &tilt_readback_buf, out_tilt_deg) {
            return false;
        }

        for i in 0..n {
            axes_io[i * 3] = axes_vec4[i * 4];
            axes_io[i * 3 + 1] = axes_vec4[i * 4 + 1];
            axes_io[i * 3 + 2] = axes_vec4[i * 4 + 2];
        }
        true
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn read_back_buffer_f32(device: &wgpu::Device, buf: &wgpu::Buffer, dst: &mut [f32]) -> bool {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    if rx.recv().ok().and_then(|r| r.ok()).is_none() {
        return false;
    }
    let data = slice.get_mapped_range();
    let floats: &[f32] = bytemuck::cast_slice(&data);
    if floats.len() != dst.len() {
        drop(data);
        buf.unmap();
        return false;
    }
    dst.copy_from_slice(floats);
    drop(data);
    buf.unmap();
    true
}

fn read_back_buffer_u32(device: &wgpu::Device, buf: &wgpu::Buffer, dst: &mut [u32]) -> bool {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    if rx.recv().ok().and_then(|r| r.ok()).is_none() {
        return false;
    }
    let data = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&data);
    if words.len() != dst.len() {
        drop(data);
        buf.unmap();
        return false;
    }
    dst.copy_from_slice(words);
    drop(data);
    buf.unmap();
    true
}

// ============================================================================
// FFI Entry Points Callable from C++ Backend
// ============================================================================

pub fn gpu_is_available() -> bool {
    GpuComputeEngine::global().is_some()
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_solve_eikonal_wavefronts(
    neighbors6: &[i32],
    voxel_flags: &[u32],
    slowness: &[f32],
    target_dir: &[f32],
    wf_indices: &[u32],
    wf_level_offsets: &[u32],
    wf_octant_bounds: &[u32],
    dx: f32,
    num_sweep_rounds: u32,
    use_directional_guidance: bool,
    tau_io: &mut [f32],
) -> bool {
    let Some(engine) = GpuComputeEngine::global() else {
        return false;
    };
    engine.solve_eikonal_wavefronts(
        neighbors6,
        voxel_flags,
        slowness,
        target_dir,
        wf_indices,
        wf_level_offsets,
        wf_octant_bounds,
        dx,
        num_sweep_rounds,
        use_directional_guidance,
        tau_io,
    )
}

pub fn gpu_evaluate_tpms_batch(
    positions_xyz: &[f32],
    pattern_kind: u8,
    wave_k: f32,
    out_psi: &mut [f32],
) -> bool {
    let Some(engine) = GpuComputeEngine::global() else {
        return false;
    };
    engine.evaluate_tpms_batch(positions_xyz, pattern_kind, wave_k, out_psi)
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_sweep_tool_clearance(
    candidate_pos_depth: &[f32],
    candidate_norm_tau: &[f32],
    brick_min_tau: &[f32],
    dense_sdf_tau: &[f32],
    int_params: &[i32],
    float_params: &[f32],
    profile_xz_flat: &[f32],
    out_flags: &mut [u32],
) -> bool {
    let Some(engine) = GpuComputeEngine::global() else {
        return false;
    };
    let n = out_flags.len();
    if n < 128
        || candidate_pos_depth.len() != n * 4
        || candidate_norm_tau.len() != n * 4
        || int_params.len() < 8
        || float_params.len() < 40
        || dense_sdf_tau.len() % 2 != 0
    {
        return false;
    }

    let mut profile_xz = [[0.0f32; 4]; 8];
    let prof_pairs = (profile_xz_flat.len() / 2).min(16);
    for k in 0..prof_pairs {
        let vec_idx = k / 2;
        let lane = (k & 1) * 2;
        profile_xz[vec_idx][lane] = profile_xz_flat[k * 2];
        profile_xz[vec_idx][lane + 1] = profile_xz_flat[k * 2 + 1];
    }

    let uniforms = ClearanceUniforms {
        candidate_count: n as u32,
        clearance_mode: int_params[0] as u32,
        profile_count: prof_pairs as u32,
        brick_nx: int_params[1],
        brick_ny: int_params[2],
        brick_nz: int_params[3],
        dense_nx: int_params[4],
        dense_ny: int_params[5],
        dense_nz: int_params[6],
        use_dense_grid: 1,
        pad0: 0,
        pad1: 0,
        nozzle_axis_bed: [
            float_params[0],
            float_params[1],
            float_params[2],
            float_params[3],
        ],
        travel_axis_bed: [
            float_params[4],
            float_params[5],
            float_params[6],
            float_params[7],
        ],
        lateral_axis_bed: [
            float_params[8],
            float_params[9],
            float_params[10],
            float_params[11],
        ],
        solid_bbox_min: [
            float_params[12],
            float_params[13],
            float_params[14],
            float_params[15],
        ],
        solid_bbox_max: [
            float_params[16],
            float_params[17],
            float_params[18],
            float_params[19],
        ],
        brick_origin_inv: [
            float_params[20],
            float_params[21],
            float_params[22],
            float_params[23],
        ],
        dense_origin_inv: [
            float_params[24],
            float_params[25],
            float_params[26],
            float_params[27],
        ],
        scalars0: [
            float_params[28],
            float_params[29],
            float_params[30],
            float_params[31],
        ],
        scalars1: [
            float_params[32],
            float_params[33],
            float_params[34],
            float_params[35],
        ],
        cone_rect_params0: [
            float_params[36],
            float_params[37],
            float_params[38],
            float_params[39],
        ],
        cone_rect_params1: [
            *float_params.get(40).unwrap_or(&0.0),
            *float_params.get(41).unwrap_or(&0.0),
            *float_params.get(42).unwrap_or(&0.0),
            0.0,
        ],
        profile_xz,
    };

    let cand_pos: &[[f32; 4]] = bytemuck::cast_slice(candidate_pos_depth);
    let cand_norm: &[[f32; 4]] = bytemuck::cast_slice(candidate_norm_tau);
    let dense_pairs: &[[f32; 2]] = bytemuck::cast_slice(dense_sdf_tau);

    engine.sweep_tool_clearance(
        &uniforms,
        cand_pos,
        cand_norm,
        brick_min_tau,
        dense_pairs,
        out_flags,
    )
}

pub fn gpu_smooth_and_clamp_tool_axes(
    neighbors6: &[i32],
    bed_normal_xyz: &[f32],
    local_max_tilt_rad: &[f32],
    num_passes: u32,
    axes_io: &mut [f32],
    out_tilt_deg: &mut [f32],
) -> bool {
    let Some(engine) = GpuComputeEngine::global() else {
        return false;
    };
    engine.smooth_and_clamp_tool_axes(
        neighbors6,
        bed_normal_xyz,
        local_max_tilt_rad,
        num_passes,
        axes_io,
        out_tilt_deg,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_evaluate_tpms_batch_or_fallback_matches_cpu() {
        let n = 1024usize;
        let mut positions = Vec::with_capacity(n * 3);
        for i in 0..n {
            let f = i as f32 * 0.13;
            positions.push(f);
            positions.push(f * 0.7 - 1.2);
            positions.push(f * 1.1 + 0.5);
        }
        let wave_k = 0.85f32;
        for pattern_kind in [2u8, 3u8, 4u8] {
            let mut out_psi = vec![0.0f32; n];
            let used_gpu =
                gpu_evaluate_tpms_batch(&positions, pattern_kind, wave_k, &mut out_psi);
            if used_gpu {
                for i in 0..n {
                    let kx = wave_k * positions[i * 3];
                    let ky = wave_k * positions[i * 3 + 1];
                    let kz = wave_k * positions[i * 3 + 2];
                    let expected = match pattern_kind {
                        2 => kx.sin() * ky.cos() + ky.sin() * kz.cos() + kz.sin() * kx.cos(),
                        3 => {
                            kx.sin() * ky.sin() * kz.sin()
                                + kx.sin() * ky.cos() * kz.cos()
                                + kx.cos() * ky.sin() * kz.cos()
                                + kx.cos() * ky.cos() * kz.sin()
                        }
                        _ => kx.cos() + ky.cos() + kz.cos(),
                    };
                    assert!(
                        (out_psi[i] - expected).abs() < 1.0e-4,
                        "TPMS pattern {pattern_kind} mismatch at {i}: got {}, expected {expected}",
                        out_psi[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_gpu_smooth_and_clamp_tool_axes_or_fallback() {
        let n = 1024usize;
        let mut neighbors6 = vec![-1i32; n * 6];
        for i in 0..n {
            if i > 0 {
                neighbors6[i * 6] = (i - 1) as i32;
            }
            if i + 1 < n {
                neighbors6[i * 6 + 1] = (i + 1) as i32;
            }
        }
        // Query a non-trivial tilted bed normal to verify AGENTS.md Rule 1 compliance
        let bed_normal = [0.0f32, 0.25881905f32, 0.9659258f32]; // 15 deg tilted frame
        let max_tilt = vec![30.0f32.to_radians(); n];
        let mut axes = Vec::with_capacity(n * 3);
        for i in 0..n {
            let angle = (i as f32 * 0.02).sin() * 45.0f32.to_radians();
            axes.push(angle.sin());
            axes.push(0.0);
            axes.push(angle.cos());
        }
        let mut out_tilt_deg = vec![0.0f32; n];
        let used_gpu = gpu_smooth_and_clamp_tool_axes(
            &neighbors6,
            &bed_normal,
            &max_tilt,
            2,
            &mut axes,
            &mut out_tilt_deg,
        );
        if used_gpu {
            for i in 0..n {
                let ax = axes[i * 3];
                let ay = axes[i * 3 + 1];
                let az = axes[i * 3 + 2];
                let len = (ax * ax + ay * ay + az * az).sqrt();
                assert!((len - 1.0).abs() < 1.0e-3);
                assert!(out_tilt_deg[i] <= 30.05);
            }
        }
    }

    #[test]
    fn test_gpu_solve_eikonal_wavefronts_or_fallback() {
        let nx = 14usize;
        let ny = 14usize;
        let nz = 12usize;
        let n = nx * ny * nz; // 2352 >= 2048
        let dx = 0.5f32;

        let idx_of = |x: usize, y: usize, z: usize| -> usize { z * (nx * ny) + y * nx + x };
        let mut neighbors6 = vec![-1i32; n * 6];
        let mut voxel_flags = vec![1u32; n]; // bit 0 = isInterior
        let slowness = vec![1.0f32; n];
        // Tilted target direction in local NozzleFrame (AGENTS.md Rule 1)
        let mut target_dir = Vec::with_capacity(n * 3);
        let mut tau_io = vec![1.0e6f32; n];

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = idx_of(x, y, z);
                    if x > 0 {
                        neighbors6[i * 6] = idx_of(x - 1, y, z) as i32;
                    }
                    if x + 1 < nx {
                        neighbors6[i * 6 + 1] = idx_of(x + 1, y, z) as i32;
                    }
                    if y > 0 {
                        neighbors6[i * 6 + 2] = idx_of(x, y - 1, z) as i32;
                    }
                    if y + 1 < ny {
                        neighbors6[i * 6 + 3] = idx_of(x, y + 1, z) as i32;
                    }
                    if z > 0 {
                        neighbors6[i * 6 + 4] = idx_of(x, y, z - 1) as i32;
                    }
                    if z + 1 < nz {
                        neighbors6[i * 6 + 5] = idx_of(x, y, z + 1) as i32;
                    }
                    if z == 0 {
                        voxel_flags[i] |= 2u32; // seed
                        tau_io[i] = 0.0;
                    }
                    target_dir.extend_from_slice(&[0.0, 0.0, 1.0]);
                }
            }
        }

        // Build 4 octant wavefront schedules
        let signs = [(1i32, 1i32), (-1, 1), (1, -1), (-1, -1)];
        let mut wf_indices = Vec::with_capacity(n * 4);
        let mut wf_level_offsets = Vec::new();
        let mut wf_octant_bounds = [0u32; 5];

        for (q, &(sx, sy)) in signs.iter().enumerate() {
            wf_octant_bounds[q] = wf_level_offsets.len() as u32;
            let idx_base = wf_indices.len() as u32;
            let mut buckets: std::collections::BTreeMap<i32, Vec<u32>> =
                std::collections::BTreeMap::new();
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let lvl = sx * (x as i32) + sy * (y as i32) + (z as i32);
                        buckets.entry(lvl).or_default().push(idx_of(x, y, z) as u32);
                    }
                }
            }
            let mut local_off = 0u32;
            wf_level_offsets.push(idx_base + local_off);
            for (_lvl, verts) in buckets {
                local_off += verts.len() as u32;
                wf_level_offsets.push(idx_base + local_off);
                wf_indices.extend(verts);
            }
        }
        wf_octant_bounds[4] = wf_level_offsets.len() as u32;

        let used_gpu = gpu_solve_eikonal_wavefronts(
            &neighbors6,
            &voxel_flags,
            &slowness,
            &target_dir,
            &wf_indices,
            &wf_level_offsets,
            &wf_octant_bounds,
            dx,
            2,
            false,
            &mut tau_io,
        );
        if used_gpu {
            for z in 0..nz {
                let expected = (z as f32) * dx;
                let got = tau_io[idx_of(nx / 2, ny / 2, z)];
                assert!(
                    (got - expected).abs() < 1.0e-3,
                    "Eikonal tau at z={z}: got {got}, expected {expected}"
                );
            }
        }
    }
}
