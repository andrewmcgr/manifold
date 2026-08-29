
# Unified Thermodynamic & Non-Newtonian Fluid Slicing Engine Architecture

This document outlines a mathematically rigorous framework that unifies **Pressure Advance (PA)**, **Retraction Length**, **Thermal Ooze**, **Part Cooling Forced Convective Gradients**, and **Non-Newtonian Shear-Thinning** into a single, cohesive state-space pipeline for 3D printer slicing engines targeting Klipper firmware.

---

## 1. Core Physics & Direct Pipeline Mapping

Instead of treating Retraction, Pressure Advance, and Temperature Compensation as disconnected user-input parameters, the entire system is governed by a single state variable: the **viscosity and elastic storage of the polymer melt zone**.

*   **Continuous Domain**: Target Extrusion/Speed Vector ➔ Volumetric Flow Rate (Q) ➔ Non-Newtonian Shear Scaling ➔ Target Base PA Value.
*   **Thermal Domain**: Target Fan Speed Percentage (F) ➔ Forced Convection Δ T ➔ Arrhenius Viscosity Shift ➔ Dynamic PA Scaling Factor.
*   **Discrete Mapped Boundary**: Combining the Flow Rate and Thermal Domain outputs yields the active `C_PA` value, which directly calculates both the Continuous Extrusion Vector and the Dynamic Retraction & Ooze Compensation values at a path stop.

---

## 2. Mathematical Formulations

### A. Non-Newtonian & Fluid Core PA Scaling (Extrusion Rate Dependency)
Because polymer melts are pseudoplastic (shear-thinning) and high volumetric flow rates introduce a solid filament core plug that stiffens the mechanical system, the ideal Pressure Advance coefficient drops as volumetric flow rate increases:

\[\eta(Q) \propto Q^{n-1} \quad \text{and} \quad k_{\text{system}}(Q) \propto Q^m \implies C_{PA}(Q) = C_{PA\_zero} \cdot Q^{-\alpha}\]

*   Q: Volumetric flow rate (mm³/s) = Layer Height × Line Width × Toolhead Velocity.
*   α: Flow Sensitivity Exponent (typically 0.15 to 0.35).
*   \(C_{PA\_zero}\): Extrapolated zero-flow baseline PA coefficient.

### B. Thermal Nozzle-Tip Gradient (Part Cooling Fan Impact)
Forced convection from part cooling fans introduces a localized chilled tip. The effective fluid temperature (\(T_{\text{effective}}\)) right at the exit orifice drops non-linearly with fan speed percentage (\(F \in [0.0, 1.0]\)):

\[T_{\text{effective}} = T_{\text{block}} - \Delta T_{\text{max}} \cdot (F)^{0.6}\]

*   \(\Delta T_{\text{max}}\): The maximum physical thermal drop at 100% fan speed (typically 5°C to 12°C).

### C. Arrhenius Temperature Scaling Matrix
The dynamic time constant scaling of the system updates continuously using the temperature-viscosity translation matrix derived via the Arrhenius equation (relative to user calibration references \(T_{\text{ref}}\), \(C_{PA\_ref}\)):

\[\Delta T = T_{\text{effective}} - T_{\text{ref}}\]

\[C_{PA\_dynamic} = C_{PA}(Q) \cdot e^{k_{PA} \cdot (-\Delta T)}\]

\[\tau_{\text{ooze}} = \tau_{\text{ref}} \cdot e^{-k_{\tau} \cdot \Delta T}\]

\[L_{\text{max}} = L_{\text{max,ref}} \cdot (1 + k_{L} \cdot \Delta T)\]

*   \(k_{PA} \approx 0.02\text{/}^\circ\text{C}\), \(k_{\tau} \approx 0.03\text{/}^\circ\text{C}\), \(k_{L} \approx 0.02\text{/}^\circ\text{C}\) (Standard polymer flow sensitivity empirical values).

### D. Time-Dependent Boundary Evaluation (Ooze & Retraction)
Retraction length represents the discrete boundary constraint of the continuous PA tracking matrix. At the timestamp where toolhead velocity drops to zero (\(t_{\text{stop}}\)), the remaining residual pressure and fluid loss over a given travel move duration (\(t_{\text{travel}}\)) is solved explicitly:

\[L_{\text{residual}} = (C_{PA\_dynamic} \cdot v_{\text{end}}) + L_{\text{static}}\]

\[L_{\text{extra,prime}} = L_{\text{max}} \cdot \left(1 - e^{-\frac{t_{\text{travel}}}{\tau_{\text{ooze}}}}\right)\]

*   \(v_{\text{end}}\): The toolhead junction velocity exiting the previous print path.
*   \(L_{\text{static}}\): A tiny mechanical constant (0.1mm to 0.3mm) used to break molten plastic surface tension at the nozzle tip.

---

## 3. Implementation Logic for Coding Agents

When compiling text segments into structural code, execute the logic in this chronological sequence:

```python
import math

class FluidDynamicsEngine:
    def __init__(self, material_profile):
        self.profile = material_profile
        # Contains: c_pa_low, low_q, c_pa_high, high_q, t_ref_c, tau_ref, l_max_ref, dt_max_fan

        # Pre-calculate Non-Newtonian Power-Law Constants from 2-point input data
        self.alpha = - (math.log(self.profile['c_pa_high']) - math.log(self.profile['c_pa_low'])) / \
                     (math.log(self.profile['high_q']) - math.log(self.profile['low_q']))
        self.c_pa_zero = self.profile['c_pa_low'] / math.pow(self.profile['low_q'], -self.alpha)

    def calculate_gcode_parameters(self, target_h, target_w, toolhead_v, fan_speed_pct, travel_time=None):
        # 1. Compute Volumetric Flow
        Q = target_h * target_w * toolhead_v

        # 2. Extract Base Shear-Thinned PA
        base_pa = self.c_pa_zero * math.pow(max(0.1, Q), -self.alpha)

        # 3. Apply Localized Convective Thermal Gradients
        t_effective = self.profile['t_block_c'] - (self.profile['dt_max_fan'] * math.pow(fan_speed_pct, 0.6))
        dT = t_effective - self.profile['t_ref_c']

        # 4. Final Continuous Domains Calculation
        live_pa = base_pa * math.exp(0.02 * (-dT))

        # 5. Handle Discrete Traveling Domains (Boundary States)
        if travel_time is not None:
            l_residual = (live_pa * toolhead_v) + self.profile.get('l_static', 0.2)

            tau_scaled = self.profile['tau_ref'] * math.exp(-0.03 * dT)
            l_max_scaled = self.profile['l_max_ref'] * (1.0 + (0.02 * dT))
            l_extra_prime = l_max_scaled * (1.0 - math.exp(-travel_time / max(0.1, tau_scaled)))

            return live_pa, l_residual, l_extra_prime

        return live_pa, 0.0, 0.0
```

---

## 4. Klipper Native G-code Output Target Syntax

When streaming output vectors, translate variables directly into Klipper native hardware macros rather than printing static `G1 E` values. This utilizes Klipper's fine lookahead engine seamlessly.

```gcode
; --- Feature Transition Matrix (Outer Perimeters to High Speed Infill) ---
M106 S0                          ; Turn off part cooling fan
SET_PRESSURE_ADVANCE ADVANCE=0.034   ; Shear-thinned high-flow optimization injected

G1 X120 Y50 E1.432 F14400        ; High-velocity continuous printing vector

; --- Line Boundary Stop Initiated ---
SET_RETRACTION RETRACT_LENGTH=0.38 RETRACT_SPEED=45
G10                              ; Execute firmware retraction mapped to computed L_residual

; --- Travel Move Executed ---
G0 X200 Y200 F18000              ; Slicer tracking determines travel duration = 1.4 seconds

; --- Feature Restart Matrix (Transitioning to Cooled Wall) ---
M106 S255                        ; Fan 100% spin up loop engaged
; Slicer calculated 1.4 second ooze at 100% fan speed requires total unretract of 0.62mm
SET_RETRACTION UNRETRACT_LENGTH=0.62 UNRETRACT_SPEED=40
G11                              ; Execute proactive, non-blobbing re-prime step
SET_PRESSURE_ADVANCE ADVANCE=0.045   ; Localized high-viscosity PA increase applied
G1 X250 Y200 E2.115 F4800        ; Resuming continuous stream geometry
```

