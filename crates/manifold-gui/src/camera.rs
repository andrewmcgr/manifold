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
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: DVec3::ZERO,
            distance: 200.0,
            yaw: std::f64::consts::FRAC_PI_4,
            pitch: std::f64::consts::FRAC_PI_6,
            fov_y_radians: 45.0_f32.to_radians(),
            near: 0.1,
            far: 10_000.0,
        }
    }
}

const MIN_PITCH: f64 = -std::f64::consts::FRAC_PI_2 + 0.01;
const MAX_PITCH: f64 = std::f64::consts::FRAC_PI_2 - 0.01;
const MIN_DISTANCE: f64 = 1.0;

impl OrbitCamera {
    /// Current eye (camera) position in world space.
    pub fn eye(&self) -> DVec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.cos();
        let y = self.distance * self.pitch.cos() * self.yaw.sin();
        let z = self.distance * self.pitch.sin();
        self.target + DVec3::new(x, y, z)
    }

    /// Apply a drag delta (in points) to orbit rotation.
    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        const SENSITIVITY: f64 = 0.01;
        self.yaw -= delta_x as f64 * SENSITIVITY;
        self.pitch = (self.pitch + delta_y as f64 * SENSITIVITY).clamp(MIN_PITCH, MAX_PITCH);
    }

    /// Apply a pan delta (in points) in the camera's local right/up plane.
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let sensitivity = self.distance * 0.0015;
        let forward = (self.target - self.eye()).normalize_or_zero();
        let right = forward.cross(DVec3::Z).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        self.target -= right * (delta_x as f64 * sensitivity);
        self.target += up * (delta_y as f64 * sensitivity);
    }

    /// Apply a scroll delta to zoom in/out.
    pub fn zoom(&mut self, delta: f32) {
        const SENSITIVITY: f64 = 0.002;
        let factor = (-delta as f64 * SENSITIVITY).exp();
        self.distance = (self.distance * factor).max(MIN_DISTANCE);
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
