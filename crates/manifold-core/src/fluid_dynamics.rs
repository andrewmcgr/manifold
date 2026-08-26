//! Unified thermodynamic and non-Newtonian fluid dynamics engine for pressure advance,
//! adaptive retraction, and thermal ooze recovery (see `RETRACTION_AND_PA.md`).

use serde::{Deserialize, Serialize};

/// Thermodynamic and non-Newtonian material configuration for fluid-driven pressure advance
/// and adaptive retraction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FluidDynamicsConfig {
    /// Low-flow calibration point: (pressure advance in seconds, volumetric flow rate in mm³/s).
    /// Default: `(0.045 s, 2.0 mm³/s)`.
    pub pa_calibration_low: (f64, f64),
    /// High-flow calibration point: (pressure advance in seconds, volumetric flow rate in mm³/s).
    /// Default: `(0.030 s, 15.0 mm³/s)`.
    pub pa_calibration_high: (f64, f64),
    /// Hotend heater block temperature in °C. Default: `240.0°C`.
    pub heater_block_temp_c: f64,
    /// Reference calibration temperature in °C where `pa_calibration` points were calibrated.
    /// Default: `240.0°C`.
    pub reference_temp_c: f64,
    /// Maximum temperature drop at the nozzle tip under 100% part cooling fan in °C.
    /// Default: `8.0°C`.
    pub max_fan_temp_drop_c: f64,
    /// Reference thermal ooze time constant in seconds. Default: `1.5 s`.
    pub ooze_time_constant_ref_s: f64,
    /// Reference maximum thermal ooze volume/length in mm of filament. Default: `0.30 mm`.
    pub ooze_max_length_ref_mm: f64,
    /// Static mechanical retraction distance to break surface tension at path stop in mm.
    /// Default: `0.15 mm`.
    pub static_retraction_mm: f64,
    /// Maximum safe retraction length in mm under dynamic fluid model. Default: `1.5 mm`.
    pub max_retraction_mm: f64,
    /// Deadband fraction for pressure advance updates (e.g. `0.10` for 10% change).
    /// Default: `0.10`.
    pub pa_deadband: f64,
}

impl Default for FluidDynamicsConfig {
    fn default() -> Self {
        Self {
            pa_calibration_low: (0.045, 2.0),
            pa_calibration_high: (0.030, 15.0),
            heater_block_temp_c: 240.0,
            reference_temp_c: 240.0,
            max_fan_temp_drop_c: 8.0,
            ooze_time_constant_ref_s: 1.5,
            ooze_max_length_ref_mm: 0.30,
            static_retraction_mm: 0.15,
            max_retraction_mm: 1.5,
            pa_deadband: 0.10,
        }
    }
}

/// Dynamic fluid dynamics state solver.
#[derive(Debug, Clone, Copy)]
pub struct FluidDynamicsEngine {
    config: FluidDynamicsConfig,
    /// Non-Newtonian shear sensitivity exponent $\alpha$.
    alpha: f64,
    /// Zero-flow baseline pressure advance coefficient $C_{\text{PA,zero}}$.
    c_pa_zero: f64,
}

impl FluidDynamicsEngine {
    /// Constructs a new fluid dynamics engine from configuration.
    #[must_use]
    pub fn new(config: FluidDynamicsConfig) -> Self {
        let (c_pa_low, q_low) = config.pa_calibration_low;
        let (c_pa_high, q_high) = config.pa_calibration_high;

        let q_low = q_low.max(0.1);
        let q_high = q_high.max(q_low + 0.1);
        let c_pa_low = c_pa_low.max(1e-4);
        let c_pa_high = c_pa_high.max(1e-4);

        // Power-law exponent: alpha = - (ln(c_pa_high) - ln(c_pa_low)) / (ln(q_high) - ln(q_low))
        let raw_alpha = -((c_pa_high.ln() - c_pa_low.ln()) / (q_high.ln() - q_low.ln()));
        let alpha = raw_alpha.clamp(0.0, 1.0);

        // Zero-flow extrapolation: c_pa_zero = c_pa_low / (q_low ^ (-alpha))
        let c_pa_zero = c_pa_low / q_low.powf(-alpha);

        Self {
            config,
            alpha,
            c_pa_zero,
        }
    }

    /// Computes the effective molten polymer temperature at the nozzle exit tip under forced fan convection.
    ///
    /// $$T_{\text{effective}} = T_{\text{block}} - \Delta T_{\text{max}} \cdot F^{0.6}$$
    #[must_use]
    pub fn effective_temperature(&self, fan_speed_fraction: f64) -> f64 {
        let fan = fan_speed_fraction.clamp(0.0, 1.0);
        self.config.heater_block_temp_c - (self.config.max_fan_temp_drop_c * fan.powf(0.6))
    }

    /// Computes the temperature deviation from the reference calibration temperature.
    ///
    /// $$\Delta T = T_{\text{effective}} - T_{\text{ref}}$$
    #[must_use]
    pub fn temperature_delta(&self, fan_speed_fraction: f64) -> f64 {
        self.effective_temperature(fan_speed_fraction) - self.config.reference_temp_c
    }

    /// Evaluates dynamic pressure advance coefficient for a given volumetric flow rate $Q$ (mm³/s)
    /// and part cooling fan speed fraction $F \in [0.0, 1.0]$.
    ///
    /// $$C_{\text{PA,dynamic}} = C_{\text{PA,zero}} \cdot Q^{-\alpha} \cdot e^{-0.02 \cdot \Delta T}$$
    #[must_use]
    pub fn dynamic_pressure_advance(&self, flow_rate_q: f64, fan_speed_fraction: f64) -> f64 {
        let q = flow_rate_q.max(0.01);
        let base_pa = self.c_pa_zero * q.powf(-self.alpha);
        let dt = self.temperature_delta(fan_speed_fraction);
        (base_pa * (-0.02 * dt).exp()).clamp(0.0, 1.0)
    }

    /// Evaluates adaptive retraction distance $L_{\text{residual}}$ (mm) at a path stop
    /// given the dynamic PA value and extruder filament exit velocity $v_{\text{filament}} = Q / A_{\text{filament}}$ (mm/s).
    ///
    /// $$L_{\text{residual}} = C_{\text{PA}} \cdot v_{\text{filament}} + L_{\text{static}} = C_{\text{PA}} \cdot \left(\frac{Q}{A_{\text{filament}}}\right) + L_{\text{static}}$$
    #[must_use]
    pub fn retraction_length(&self, dynamic_pa: f64, filament_velocity_mm_s: f64) -> f64 {
        let v = filament_velocity_mm_s.max(0.0);
        let l_residual = (dynamic_pa * v) + self.config.static_retraction_mm;
        l_residual.clamp(
            self.config.static_retraction_mm * 0.5,
            self.config.max_retraction_mm,
        )
    }

    /// Evaluates extra prime length $L_{\text{extra,prime}}$ (mm) to compensate for thermal ooze
    /// accumulated during travel duration $t_{\text{travel}}$ (seconds) at fan speed fraction $F$.
    ///
    /// $$L_{\text{extra,prime}} = L_{\text{max,scaled}} \cdot \left(1 - e^{-t_{\text{travel}} / \tau_{\text{scaled}}}\right)$$
    #[must_use]
    pub fn extra_prime_length(&self, travel_time_s: f64, fan_speed_fraction: f64) -> f64 {
        if travel_time_s <= 1e-4 {
            return 0.0;
        }
        let dt = self.temperature_delta(fan_speed_fraction);
        let tau_scaled = (self.config.ooze_time_constant_ref_s * (-0.03 * dt).exp()).max(0.01);
        let l_max_scaled = (self.config.ooze_max_length_ref_mm * (1.0 + 0.02 * dt)).max(0.0);

        let ooze = l_max_scaled * (1.0 - (-travel_time_s / tau_scaled).exp());
        ooze.clamp(0.0, 1.5)
    }

    /// Evaluates unretract distance (mm) needed after a travel move of duration $t_{\text{travel}}$
    /// following a retraction of distance $L_{\text{retract}}$.
    #[must_use]
    pub fn unretract_length(
        &self,
        retraction_len: f64,
        travel_time_s: f64,
        fan_speed_fraction: f64,
    ) -> f64 {
        retraction_len + self.extra_prime_length(travel_time_s, fan_speed_fraction)
    }

    /// Returns the active configuration.
    #[must_use]
    pub fn config(&self) -> &FluidDynamicsConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_law_fits_calibration_points_accurately() {
        let config = FluidDynamicsConfig {
            pa_calibration_low: (0.040, 2.0),
            pa_calibration_high: (0.025, 12.0),
            ..Default::default()
        };
        let engine = FluidDynamicsEngine::new(config);

        let pa_low = engine.dynamic_pressure_advance(2.0, 0.0);
        let pa_high = engine.dynamic_pressure_advance(12.0, 0.0);

        assert!((pa_low - 0.040).abs() < 1e-4, "pa_low was {pa_low}");
        assert!((pa_high - 0.025).abs() < 1e-4, "pa_high was {pa_high}");
    }

    #[test]
    fn fan_cooling_drops_tip_temperature_and_increases_viscosity_pa() {
        let config = FluidDynamicsConfig {
            heater_block_temp_c: 240.0,
            reference_temp_c: 240.0,
            max_fan_temp_drop_c: 10.0,
            ..Default::default()
        };
        let engine = FluidDynamicsEngine::new(config);

        let t_no_fan = engine.effective_temperature(0.0);
        let t_full_fan = engine.effective_temperature(1.0);

        assert_eq!(t_no_fan, 240.0);
        assert_eq!(t_full_fan, 230.0);

        let pa_no_fan = engine.dynamic_pressure_advance(5.0, 0.0);
        let pa_full_fan = engine.dynamic_pressure_advance(5.0, 1.0);

        assert!(
            pa_full_fan > pa_no_fan,
            "Cooled tip must have higher PA due to increased viscosity (was {pa_full_fan} vs {pa_no_fan})"
        );
    }

    #[test]
    fn retraction_scales_with_junction_velocity() {
        let config = FluidDynamicsConfig {
            static_retraction_mm: 0.15,
            max_retraction_mm: 1.5,
            ..Default::default()
        };
        let engine = FluidDynamicsEngine::new(config);

        let r_zero_v = engine.retraction_length(0.035, 0.0);
        let r_slow_v = engine.retraction_length(0.035, 2.0);
        let r_fast_v = engine.retraction_length(0.035, 10.0);

        assert_eq!(r_zero_v, 0.15);
        assert!((r_slow_v - (0.15 + 0.035 * 2.0)).abs() < 1e-5);
        assert!(r_fast_v > r_slow_v);
    }

    #[test]
    fn thermal_ooze_approaches_maximum_asymptotically_with_travel_time() {
        let config = FluidDynamicsConfig {
            ooze_time_constant_ref_s: 2.0,
            ooze_max_length_ref_mm: 0.40,
            ..Default::default()
        };
        let engine = FluidDynamicsEngine::new(config);

        let prime_instant = engine.extra_prime_length(0.0, 0.0);
        let prime_short = engine.extra_prime_length(0.5, 0.0);
        let prime_long = engine.extra_prime_length(10.0, 0.0);

        assert_eq!(prime_instant, 0.0);
        assert!(prime_short > 0.0);
        assert!(prime_long > prime_short);
        assert!((prime_long - 0.40).abs() < 0.01);
    }

    #[test]
    fn thermal_ooze_supports_fast_50ms_time_constant() {
        let config = FluidDynamicsConfig {
            ooze_time_constant_ref_s: 0.05, // 50 ms
            ooze_max_length_ref_mm: 0.20,
            ..Default::default()
        };
        let engine = FluidDynamicsEngine::new(config);

        // At t = 50ms (1 time constant), ooze should be ~63.2% of max (0.20 * 0.632 = 0.126)
        let prime_1tau = engine.extra_prime_length(0.05, 0.0);
        assert!((prime_1tau - 0.126).abs() < 0.01);

        // At t = 200ms (4 time constants), ooze should approach ~98% of max (0.20 * 0.98 = 0.196)
        let prime_4tau = engine.extra_prime_length(0.20, 0.0);
        assert!((prime_4tau - 0.196).abs() < 0.01);
    }

    #[test]
    fn deserialization_fills_missing_fields_from_defaults() {
        let json = r#"{
            "pa_calibration_low": [0.040, 2.0],
            "pa_calibration_high": [0.025, 12.0]
        }"#;
        let config: FluidDynamicsConfig =
            serde_json::from_str(json).expect("deserialization with missing fields must succeed");
        assert_eq!(config.pa_calibration_low, (0.040, 2.0));
        assert_eq!(config.max_retraction_mm, 1.5);
        assert_eq!(config.static_retraction_mm, 0.15);
    }
}
