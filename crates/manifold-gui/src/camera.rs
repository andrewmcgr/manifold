//! Orbit camera: drag-to-rotate, scroll-to-zoom, driven by pointer input
//! in the viewport (Phase 5, see ROADMAP.md).

use glam::{DMat4, DVec3, Mat4, Vec3};

/// An orbit camera looking at `target` from `distance` away, at
/// `yaw`/`pitch` angles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    pub target: DVec3,
    pub distance: f64,
    /// Rotation around the world Z axis (radians).
    pub yaw: f64,
    /// Rotation up/down from the horizontal plane (radians).
    pub pitch: f64,
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
    pub min_distance: f64,
    pub max_distance: f64,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: DVec3::ZERO,
            distance: 200.0,
            yaw: -std::f64::consts::FRAC_PI_4,
            pitch: std::f64::consts::FRAC_PI_6,
            fov_y_radians: 45.0_f32.to_radians(),
            near: 0.01,
            far: 10_000.0,
            min_distance: MIN_DISTANCE,
            max_distance: DEFAULT_MAX_DISTANCE,
        }
    }
}

const MIN_PITCH: f64 = -std::f64::consts::FRAC_PI_2 + 0.01;
const MAX_PITCH: f64 = std::f64::consts::FRAC_PI_2 - 0.01;
pub const MIN_DISTANCE: f64 = 0.05;
pub const DEFAULT_MAX_DISTANCE: f64 = 2000.0;

impl OrbitCamera {
    /// Current eye (camera) position in world space.
    pub fn eye(&self) -> DVec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.cos();
        let y = self.distance * self.pitch.cos() * self.yaw.sin();
        let z = self.distance * self.pitch.sin();
        self.target + DVec3::new(x, y, z)
    }

    /// Apply a drag delta (in points) to orbit rotation around [`Self::target`].
    #[allow(dead_code)]
    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        self.orbit_around(self.target, delta_x, delta_y);
    }

    /// Orbit the camera around an arbitrary world-space `pivot` point by
    /// `delta_x` (yaw) and `delta_y` (pitch).
    ///
    /// - If `pivot == self.eye()`, this rotates the camera in place (first-person look).
    /// - If `pivot == self.target`, this is standard orbit around target.
    /// - For any surface hit point, this orbits around that surface point without
    ///   snapping the point to the center of the screen or jumping the view on drag start.
    pub fn orbit_around(&mut self, pivot: DVec3, delta_x: f32, delta_y: f32) {
        if delta_x == 0.0 && delta_y == 0.0 {
            return;
        }
        const SENSITIVITY: f64 = 0.01;
        let eye = self.eye();
        let forward = (self.target - eye).normalize_or_zero();
        let right = forward.cross(DVec3::Z).normalize_or_zero();

        let q_yaw = glam::DQuat::from_axis_angle(DVec3::Z, -delta_x as f64 * SENSITIVITY);
        let q_pitch = glam::DQuat::from_axis_angle(right, delta_y as f64 * SENSITIVITY);
        let q = q_yaw * q_pitch;

        let to_eye = eye - pivot;
        let to_target = self.target - pivot;

        let new_eye = pivot + q * to_eye;
        let new_target = pivot + q * to_target;

        let new_forward = (new_target - new_eye).normalize_or_zero();
        let new_pitch = (-new_forward.z)
            .clamp(-1.0, 1.0)
            .asin()
            .clamp(MIN_PITCH, MAX_PITCH);
        let new_yaw = (-new_forward.y).atan2(-new_forward.x);

        self.pitch = new_pitch;
        self.yaw = new_yaw;

        let offset = DVec3::new(
            self.distance * self.pitch.cos() * self.yaw.cos(),
            self.distance * self.pitch.cos() * self.yaw.sin(),
            self.distance * self.pitch.sin(),
        );
        self.target = new_eye - offset;
    }

    /// Apply a pan delta (in points) in the camera's local right/up plane,
    /// scaled to match `viewport_height` at depth `depth` so screen motion tracks 1:1.
    pub fn pan_with_depth(&mut self, delta_x: f32, delta_y: f32, viewport_height: f32, depth: f64) {
        let half_fov = (self.fov_y_radians as f64 * 0.5).tan();
        let units_per_point = 2.0 * depth * half_fov / (viewport_height.max(1.0) as f64);
        let forward = (self.target - self.eye()).normalize_or_zero();
        let right = forward.cross(DVec3::Z).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        self.target -= right * (delta_x as f64 * units_per_point);
        self.target += up * (delta_y as f64 * units_per_point);
    }

    /// Apply a pan delta (in points) in the camera's local right/up plane at [`Self::distance`] depth.
    #[allow(dead_code)]
    pub fn pan(&mut self, delta_x: f32, delta_y: f32, viewport_height: f32) {
        self.pan_with_depth(delta_x, delta_y, viewport_height, self.distance);
    }

    /// Retarget the camera to `new_target` while keeping [`Self::eye`]
    /// in the exact same world-space position. Updates `distance`, `yaw`,
    /// and `pitch` accordingly.
    #[allow(dead_code)]
    pub fn set_target_preserving_eye(&mut self, new_target: DVec3) {
        let eye = self.eye();
        let to_eye = eye - new_target;
        let new_distance = to_eye.length();
        if new_distance < 1e-6 {
            return;
        }
        self.target = new_target;
        self.distance = new_distance.clamp(self.min_distance, self.max_distance);

        let pitch = (to_eye.z / new_distance).clamp(-1.0, 1.0).asin();
        let yaw = to_eye.y.atan2(to_eye.x);

        self.pitch = pitch.clamp(MIN_PITCH, MAX_PITCH);
        self.yaw = yaw;
    }

    /// Rotate the camera in place around its current [`Self::eye`] position
    /// (first-person look), keeping `eye` fixed and moving `target`.
    #[allow(dead_code)]
    pub fn rotate_camera(&mut self, delta_x: f32, delta_y: f32) {
        self.orbit_around(self.eye(), delta_x, delta_y);
    }

    /// Cast a world-space ray `(origin, direction)` through a screen-space cursor
    /// position `pos` within viewport rectangle `rect`.
    pub fn unproject_ray(&self, rect: egui::Rect, pos: egui::Pos2) -> (DVec3, DVec3) {
        let aspect_ratio = rect.width() / rect.height().max(1.0);
        let view_proj = self.projection_matrix_f64(aspect_ratio) * self.view_matrix_f64();
        let inv_view_proj = view_proj.inverse();

        let ndc_x = ((pos.x - rect.min.x) / rect.width().max(1.0)) * 2.0 - 1.0;
        let ndc_y = (1.0 - (pos.y - rect.min.y) / rect.height().max(1.0)) * 2.0 - 1.0;

        let p_near = inv_view_proj.project_point3(DVec3::new(ndc_x as f64, ndc_y as f64, 0.0));
        let p_far = inv_view_proj.project_point3(DVec3::new(ndc_x as f64, ndc_y as f64, 1.0));
        let dir = (p_far - p_near).normalize_or_zero();
        (self.eye(), dir)
    }

    /// Apply a scroll delta to zoom in/out, clamped to `[min_distance, max_distance]`.
    pub fn zoom(&mut self, delta: f32) {
        const SENSITIVITY: f64 = 0.002;
        let factor = (-delta as f64 * SENSITIVITY).exp();
        self.distance = (self.distance * factor).clamp(self.min_distance, self.max_distance);
    }

    /// Re-center and re-distance the camera so the axis-aligned box
    /// `min..max` (e.g. the machine's build volume) fits in view, keeping
    /// the current `yaw`/`pitch`/`fov_y_radians`. Used to frame the whole
    /// bed on startup instead of defaulting to a view of the origin.
    ///
    /// Also updates [`Self::max_distance`] so that the scene occupies
    /// approximately 1/3 of the vertical screen height at maximum zoom-out.
    pub fn frame(&mut self, min: DVec3, max: DVec3) {
        const FIT_MARGIN: f64 = 1.3;
        self.target = (min + max) * 0.5;
        self.update_max_distance(min, max);
        let radius = (max - min).length() * 0.5;
        let half_fov = self.fov_y_radians as f64 * 0.5;
        let fit_distance = radius / half_fov.tan();
        self.distance = (fit_distance * FIT_MARGIN).clamp(self.min_distance, self.max_distance);
    }

    /// Update [`Self::max_distance`] from an axis-aligned bounding box `min..max`
    /// so that the bounding sphere occupies approximately 1/3 of the screen height
    /// at maximum zoom-out (`3.0 * radius / tan(fov_y / 2)`).
    pub fn update_max_distance(&mut self, min: DVec3, max: DVec3) {
        let radius = (max - min).length() * 0.5;
        let half_fov = self.fov_y_radians as f64 * 0.5;
        let fit_distance = radius / half_fov.tan();
        // At 3.0 * fit_distance, the scene subtends 1/3 of the vertical FOV.
        self.max_distance = (fit_distance * 3.0).max(self.min_distance);
        self.distance = self.distance.clamp(self.min_distance, self.max_distance);
    }

    /// The camera-space view matrix (world -> camera).
    pub fn view_matrix(&self) -> Mat4 {
        let eye = self.eye().as_vec3();
        let target = self.target.as_vec3();
        Mat4::look_at_rh(eye, target, Vec3::Z)
    }

    /// The projection matrix for the given viewport aspect ratio (width /
    /// height).
    pub fn projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y_radians, aspect_ratio, self.near, self.far)
    }

    /// Double-precision view matrix, for feeding into APIs (e.g.
    /// `transform-gizmo-egui`) that expect `f64` matrices to line up with
    /// `manifold-core`'s `f64` object transforms.
    pub fn view_matrix_f64(&self) -> DMat4 {
        DMat4::look_at_rh(self.eye(), self.target, DVec3::Z)
    }

    /// Double-precision projection matrix — see [`Self::view_matrix_f64`].
    pub fn projection_matrix_f64(&self, aspect_ratio: f32) -> DMat4 {
        DMat4::perspective_rh(
            self.fov_y_radians as f64,
            aspect_ratio as f64,
            self.near as f64,
            self.far as f64,
        )
    }

    /// The combined view-projection matrix for the given viewport aspect
    /// ratio (width / height).
    pub fn view_proj(&self, aspect_ratio: f32) -> Mat4 {
        self.projection_matrix(aspect_ratio) * self.view_matrix()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_centers_target_on_bounds_midpoint() {
        let mut camera = OrbitCamera::default();
        camera.frame(DVec3::ZERO, DVec3::new(200.0, 200.0, 200.0));
        assert_eq!(camera.target, DVec3::new(100.0, 100.0, 100.0));
    }

    #[test]
    fn frame_sets_distance_wide_enough_to_fit_bounds() {
        let mut camera = OrbitCamera::default();
        let min = DVec3::ZERO;
        let max = DVec3::new(200.0, 200.0, 200.0);
        camera.frame(min, max);

        let radius = (max - min).length() * 0.5;
        let half_fov = camera.fov_y_radians as f64 * 0.5;
        let fit_distance = radius / half_fov.tan();
        assert!(camera.distance > fit_distance);
    }

    #[test]
    fn frame_never_goes_below_minimum_distance_for_a_tiny_box() {
        let mut camera = OrbitCamera::default();
        camera.frame(DVec3::ZERO, DVec3::splat(1e-9));
        assert!(camera.distance >= camera.min_distance);
    }

    #[test]
    fn zoom_clamps_to_max_distance_derived_from_bounds() {
        let mut camera = OrbitCamera::default();
        let min = DVec3::ZERO;
        let max = DVec3::new(200.0, 200.0, 200.0);
        camera.frame(min, max);

        // Zoom out aggressively with large negative scroll deltas
        for _ in 0..100 {
            camera.zoom(-100.0);
        }
        assert!((camera.distance - camera.max_distance).abs() < 1e-6);

        let radius = (max - min).length() * 0.5;
        let half_fov = camera.fov_y_radians as f64 * 0.5;
        let fit_distance = radius / half_fov.tan();
        // At max_distance, it should be approximately 3.0 * fit_distance (scene is 1/3 screen)
        assert!((camera.max_distance - fit_distance * 3.0).abs() < 1e-6);
    }

    #[test]
    fn set_target_preserving_eye_keeps_eye_position_identical() {
        let mut camera = OrbitCamera::default();
        let orig_eye = camera.eye();
        let new_target = DVec3::new(50.0, 30.0, 20.0);
        camera.set_target_preserving_eye(new_target);
        let new_eye = camera.eye();
        assert!(
            (new_eye - orig_eye).length() < 1e-6,
            "eye drifted from {:?} to {:?}",
            orig_eye,
            new_eye
        );
        assert_eq!(camera.target, new_target);
    }

    #[test]
    fn orbit_around_with_zero_delta_leaves_camera_completely_unchanged() {
        let mut camera = OrbitCamera::default();
        let orig_eye = camera.eye();
        let orig_target = camera.target;
        let orig_yaw = camera.yaw;
        let orig_pitch = camera.pitch;

        let pivot = DVec3::new(45.0, -120.0, 15.0);
        camera.orbit_around(pivot, 0.0, 0.0);

        assert!((camera.eye() - orig_eye).length() < 1e-12);
        assert!((camera.target - orig_target).length() < 1e-12);
        assert!((camera.yaw - orig_yaw).abs() < 1e-12);
        assert!((camera.pitch - orig_pitch).abs() < 1e-12);
    }

    #[test]
    fn orbit_around_surface_point_rotates_around_pivot() {
        let mut camera = OrbitCamera::default();
        let pivot = DVec3::new(50.0, 50.0, 0.0);
        let orig_dist_to_pivot = (camera.eye() - pivot).length();

        camera.orbit_around(pivot, 10.0, 5.0);

        let new_dist_to_pivot = (camera.eye() - pivot).length();
        assert!(
            (new_dist_to_pivot - orig_dist_to_pivot).abs() < 1e-6,
            "distance to pivot changed from {} to {}",
            orig_dist_to_pivot,
            new_dist_to_pivot
        );
    }

    #[test]
    fn pan_with_depth_does_not_change_yaw_or_pitch() {
        let mut camera = OrbitCamera::default();
        let orig_yaw = camera.yaw;
        let orig_pitch = camera.pitch;

        camera.pan_with_depth(50.0, -30.0, 800.0, 150.0);

        assert_eq!(camera.yaw, orig_yaw);
        assert_eq!(camera.pitch, orig_pitch);
    }

    #[test]
    fn pan_with_viewport_height_produces_exact_screen_scale_motion() {
        let mut camera = OrbitCamera {
            distance: 100.0,
            ..Default::default()
        };
        let orig_target = camera.target;
        let viewport_h = 1000.0;
        camera.pan(0.0, 100.0, viewport_h);

        let half_fov = (camera.fov_y_radians as f64 * 0.5).tan();
        let expected_units = 2.0 * 100.0 * half_fov / viewport_h as f64 * 100.0;
        let moved = (camera.target - orig_target).length();
        assert!((moved - expected_units).abs() < 1e-6);
    }
}
