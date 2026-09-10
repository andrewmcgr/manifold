//! Build version metadata derived from git describe at compile time.

/// Build version string, e.g. `"v0.4.0"` or `"v0.4.0-14-g9369956"`.
pub const MANIFOLD_VERSION: &str = env!("MANIFOLD_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifold_version_is_non_empty_and_starts_with_v() {
        assert!(!MANIFOLD_VERSION.is_empty());
        assert!(MANIFOLD_VERSION.starts_with('v'));
    }
}
