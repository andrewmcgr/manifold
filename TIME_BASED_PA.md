# Task: Implement Time-Based Dynamic Residual Pressure Flow Compensation

## Context
We are implementing an advanced geometry post-processing feature for our 3D printing slicer. Standard slicers use simple spatial lookup tables (like OrcaSlicer's Small Area Flow Compensation) to reduce flow on short paths, which fails to account for velocity variations. 

Because our slicer operates in true post-extrusion volume and has reliable segment execution time estimates, we want to implement a physics-based **Transient Nozzle Pressure Tracking System**. This system will model the hotend melt zone as a first-order differential system to dynamically adjust extrusion volumes based on accumulated residual pressure.

## Objective
Write a robust, self-contained Python module (or integrate into our existing G-code geometry processing pipe) that tracks internal nozzle pressure `P` across consecutive toolpath segments and modifies the commanded extrusion volume `V_nominal` accordingly.

## Core Physics & Mathematical Model
The melt zone is modeled using our Pressure Advance constant ($K$, in seconds). The internal nozzle pressure $P(t)$ updates across a segment of duration $t_{move}$ based on the target volumetric flow rate $Q_{target} = V_{nominal} / t_{move}$.

### 1. State Update Equation
For a segment starting with pressure $P_{start}$, the final pressure $P_{end}$ at the end of the move is:
$$P_{end} = Q_{target} + (P_{start} - Q_{target}) \cdot e^{-\frac{t_{move}}{K}}$$

### 2. Average Pressure Equation
The average pressure $P_{average}$ acting on the melt zone during the move duration is found by integrating $P(t)$:
$$P_{average} = Q_{target} + \frac{K}{t_{move}} \cdot (P_{start} - Q_{target}) \cdot \left(1 - e^{-\frac{t_{move}}{K}}\right)$$

### 3. Compensation Modifier ($M$)
If the average pressure is higher than the requested target flow ($P_{average} > Q_{target}$), it indicates the nozzle is pre-pressurized from previous rapid movements. We capitalize on this residual pressure by reducing input volume:
$$M = \frac{1.0}{\left(\frac{P_{average}}{Q_{target}}\right)} = \frac{Q_{target}}{P_{average}}$$
*   **Safety Bound:** $M$ must be clamped to a user-defined minimum threshold `M_min` (default `0.75`) to prevent total starvation.
*   If $P_{average} \le Q_{target}$, no reduction is applied ($M = 1.0$).

## Requirements for the Implementation

1.  **Stateful Tracker Class:** Create a `PressureTracker` class that maintains `P_current` across sequential segment executions.
2.  **Segment Processing Logic:**
    *   **Extrusion Moves ($V_{nominal} > 0$):** Calculate $Q_{target}$, solve for $P_{average}$, compute the multiplier $M$, apply it to find $V_{compensated}$, and then update `P_current` using the *new compensated flow rate* ($Q_{compensated} = V_{compensated} / t_{move}$).
    *   **Travel Moves ($V_{nominal} == 0$):** Decay `P_current` exponentially over $t_{move}$ toward $0$. Handle the edge case where $t_{move} == 0$ cleanly.
    *   **Retractions ($V_{nominal} < 0$):** Allow negative flow rates to model pressure drops or tension inside the hotend melt zone, but ensure the tracking values remain physically sane (e.g., clamp negative pressure to a reasonable floor if needed to prevent infinite loop errors).
3.  **Code Quality:**
    *   Include descriptive comments referencing the math above.
    *   Add comprehensive unit tests simulating a rapid zig-zag infill path (e.g., ten 0.5mm movements at 100mm/s) to verify that pressure accumulates and the flow multiplier gradually decreases across successive steps.

## Expected Output
An implementation plan for this feature in Manifold, including documentation based on this file.
