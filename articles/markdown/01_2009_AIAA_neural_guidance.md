# Neural Networks as a Guidance Solution for Soft-Landing and Aerocapture

> **Faithful extract** of the foundational paper. This is the direct predecessor of the new article.
> Reuse: problem formulation, NN bank-angle parameterization, GA training, the FTC baseline, and the FTC-vs-NN result tables.

**Authors:** G. Gelly, P. Vernis — ASTRIUM Space Transportation, 78130 Les Mureaux, France
**Venue:** AIAA Guidance, Navigation, and Control Conference, Chicago, Illinois, August 2009
**Pages:** 21

---

## Abstract (verbatim)

This paper presents guidance algorithms based on neural networks and illustrates their performance for both aerocapture and soft-landing applications. When tackling guidance problems that do not admit a complete analytic solution, this neural network approach makes it easier to determine a satisfactory command law without making strong simplifying assumptions. Thanks to genetic algorithms, we successfully trained feed-forward neural networks with one hidden layer for both missions. And using comprehensive simulation tools, we then performed Monte Carlo analyses to compare our algorithm with classic guidance methods (Apollo E guidance for soft-landing and an extension of the Cerimele-Gamble scheme for aerocapture). The results show that neural networks can be an interesting alternative as a more optimal guidance scheme.

---

## Nomenclature

**Acronyms:** DOF (Degree of Freedom), EI (Entry Interface), FTC (Feedback Trajectory Control), GA (Genetic Algorithms), GNC (Guidance, Navigation and Control), MSR (Mars Sample Return), OOP (Out-Of-Plane), PID (Proportional, Integral and Derivative), RCS (Reaction Control System).

**Symbols:** $t$ time [s], $t_{go}$ time-to-go [s], $m$ mass [kg], $\dot m$ mass flow rate [kg/s], $I_{sp}$ specific impulse [s], $g_0$ propulsive constant [m/s²], $g$ gravitational acceleration [m/s²], $\rho$ atmospheric density [kg/m³], $\mu$ bank angle [rad], $q$ dynamic pressure [Pa], $h$ altitude [m], $Z_a$ apoapsis altitude [km], $Z_p$ periapsis altitude [km], $i$ orbit inclination [deg], $G$ gain, $\Delta V$ propulsive correction cost [m/s].

---

## I. Introduction — key framing (reusable)

- Neural networks are "especially valuable to handle problems with no analytic solutions. This is typically the case when confronted to the guidance of an atmospheric flight."
- The classic move — "one makes strong simplifications to be able to determine a satisfactory command law" — "usually lead[s] to a non-optimal resolution of the guidance problem."
- Architecture choice: "a trade-off on various neural networks architectures led us to consider feed-forward neural networks with one hidden layer."
- Training: "the best candidate was based on genetic algorithms which provide a powerful search process."
- The "neural guidance" handles the *complete* guidance problem end-to-end:
  - soft-landing: progressive braking + lateral maneuvering;
  - aerocapture: bank-angle management to minimize insertion cost on a targeted circular orbit.

---

## II. Neural Networks

### II.A. Architecture

Fully connected feed-forward network with **one hidden layer**. To each input $p \in \mathbb{R}^q$ it associates an output $z \in \mathbb{R}^s$:

$$z = \sigma_2\!\left(W_2 \cdot \sigma_1\!\left(W_1 \cdot p + b_1\right) + b_2\right) \tag{1}$$

- $W_1 \in \mathbb{R}^{r\times q}$, $W_2 \in \mathbb{R}^{s\times r}$ — interconnection (weight) matrices;
- $b_1 \in \mathbb{R}^r$, $b_2 \in \mathbb{R}^s$ — bias vectors;
- $\sigma_1, \sigma_2$ — transfer functions, "typically chosen among bounded non-linear functions such as the hyperbolic tangent."
- Hidden layer $L_1=(\sigma_1,W_1,b_1)$, output layer $L_2=(\sigma_2,W_2,b_2)$.

Per-component form:

$$h_j = \sigma_1\!\left(\textstyle\sum_i W_{1,ji}\, p_i + b_{1,j}\right), \qquad z_k = \sigma_2\!\left(\textstyle\sum_j W_{2,kj}\, h_j + b_{2,k}\right)$$

### II.B. Training — Genetic Algorithms

> "Since the goal of a neural network guidance is not to mimic the behavior of another guidance scheme, a training technique has been developed based on genetic algorithms... The main advantage of this training technique is that it does not require any sort of predetermined trajectories (or pair input/output) to solve the guidance problem."

GA features as implemented in the study:
- Each individual = a neural network **encoded as a real-valued vector of weights and biases**. All individuals share the same transfer function (tanh).
- Genetic operators: reproduction (crossing of two individuals) + mutation (random modification of one gene).
- **Subpopulations** evolve independently with a **migration scheme**: the best individual of a subpopulation is duplicated into the next one.
- Parents and children compete → best parents survive ("prevents a good individual from being lost by the probabilistic nature of reproduction").
- Fitness = simulated mission performance under **both nominal and dispersed conditions**, with simplified models to bound compute.
- "To avoid heavy computations and any problem due to a low amount of simulation cases, **we regularly change the set of dispersed conditions during the training process**." *(This is the ancestor of the repo's rotating/adaptive seed strategy.)*

---

## III. Soft-Landing (summary — secondary for the new article)

3-phase Martian powered descent after parachute jettison: (1) cancel wind-induced lateral velocity, (2) choose a landing site within a 40° field of view, (3) precision landing with possible retargeting.

**Nominal initial conditions:** altitude 2 km (first target designation at 1.2 km), relative velocity 80 m/s, FPA −90°, initial mass 560 kg, thrust 12×300 N (throttle 20–100%), $I_{sp}$ 280 s.
**Touchdown requirements:** vertical velocity < 3 m/s, horizontal velocity < 1 m/s, angular rates < 2.5 deg/s, tilt < 5°.
**Testbed:** 6-DOF, guidance @ 10 Hz, 3-axis PID control @ 20 Hz. MC = 1000 trajectories.

### Classic baseline: Apollo E guidance (Cherry, 1964)

Command (propulsive acceleration) is a polynomial in time; order > 2 gives negligible benefit for Mars descent. With two boundary conditions (final position + velocity):

$$u(t) = C_1 + C_2\,(t_{go} - t) \tag{2}$$
$$a(t) = u(t) + g \tag{3}$$
$$v(t) = C_1 t + C_2\!\left(t_{go}\,t - \tfrac{1}{2}t^2\right) + g t + v_0 \tag{4}$$
$$p(t) = \tfrac{1}{2}C_1 t^2 + C_2\!\left(\tfrac{1}{2}t_{go}\,t^2 - \tfrac{1}{6}t^3\right) + \tfrac{1}{2}g t^2 + v_0 t + p_0 \tag{5}$$

Coefficients to satisfy the targeted state at $t=t_{go}$:

$$C_1 = \frac{2}{t_{go}}\left(2 v_{targ} + v_0\right) - \frac{6}{t_{go}^2}\left(p_{targ} - p_0\right) - g \tag{6}$$
$$C_2 = -\frac{6}{t_{go}^2}\left(v_{targ} + v_0\right) + \frac{12}{t_{go}^3}\left(p_{targ} - p_0\right) \tag{7}$$

Time-to-go from an iterative process (rocket equation), with $g_0 = 9.80665$ m/s²:

$$t_{go} = \frac{m}{\dot m}\left(1 - e^{-\Delta V / (g_0 I_{sp})}\right) \tag{8}, \qquad \Delta V = \left\| v_{targ} - (v_0 + g\,t_{go}) \right\| \tag{9}$$

Because terms divide by $t_{go}$, the command is frozen at its previous value when $t_{go}$ gets small.

### Neural soft-landing scheme

Two feed-forward networks:
- **Thrust-magnitude net:** 4 inputs (altitude, distance to target, vertical velocity, horizontal-velocity magnitude) → 8 hidden → 1 output (commanded thrust).
- **Tilt net:** 4 inputs (position offset and horizontal velocity on the considered axis, altitude, vertical velocity) → 6 hidden → 1 output (commanded tilt). Called **twice per cycle** (one per horizontal axis).

Two distinct trainings via differently-shaped fitness functions → **"low consumption"** and **"low tilt"** policies. 600 generations, two populations of 20, mutation rate 0.01, 1000 dispersed sims (disjoint from the 1000 MC comparison cases). Both fitnesses strongly favor final vertical velocity < 2.5 m/s and no burnout (< 80 kg fuel).

**Headline result (worst values at touchdown, vs Apollo):**

| Performance | Apollo | NN "low consumption" | NN "low tilt" | Req. |
|---|---|---|---|---|
| Final position offset (m) | 9.68 | 2.10 (−78%) | 2.23 (−77%) | – |
| Final lateral velocity (m/s) | 1.15 | 0.37 (−68%) | 0.31 (−73%) | 1.0 |
| Final vertical velocity (m/s) | 2.70 | 2.53 (−6%) | 2.55 (−6%) | 3.0 |
| Final angular rate (deg/s) | 2.06 | 1.91 (−7%) | 0.15 (−93%) | 2.5 |
| Final tilt (deg) | 4.82 | 2.08 (−57%) | 1.11 (−77%) | 5.0 |
| Max tilt during descent (deg) | 62.65 | 58.51 (−7%) | 39.50 (−37%) | – |
| Consumption (kg) | 77.20 | 50.19 (−35%) | 75.40 (−2%) | – |

Takeaway: the **fitness function shapes the behavior** — same architecture yields a fuel-optimal or a visibility-optimal policy. Neural guidance beats Apollo on accuracy at lower cost.

---

## IV. Aerocapture (PRIMARY for the new article)

> "Aerocapture is a propulsion-free insertion from a hyperbolic interplanetary trip into an elliptical orbit around a planet surrounded by an atmosphere." Energy dissipation comes from aerodynamic lift and drag while passing through the atmosphere; the only control is **bank-angle modulation**, whose authority is **proportional to dynamic pressure**.

### IV.A. Mission — robotic Mars Sample Return (MSR)

Nominal entry conditions at **120 km** altitude:
- relative velocity: **5687 m/s**
- flight path angle: **−10.24°**
- heading angle: **38.04°**

Reference orbital parameters at atmosphere exit (for a circular 500 km parking orbit):
- apoapsis: **500 km**, periapsis: **11 km**, inclination: **50°**

### IV.B. Aerocapture corridor (reusable formulation)

Characterize the mission in the **(orbital energy, dynamic pressure)** plane. The corridor is bounded by two constant-bank-angle trajectories:
- **overshoot** — limit between an elliptic-orbit exit and a hyperbolic-orbit exit;
- **undershoot** — limit between an elliptic-orbit exit and a crash.

A more practical **restricted corridor** uses constant-bank trajectories leading to apoapsis errors of $+\delta Z_a$ (overshoot) and $-\delta Z_a$ (undershoot), with $\delta Z_a = 100$ km here.

The aerocapture begins at EI with orbital energy **4.91 MJ/kg** and ends at atmosphere exit with orbital energy **−5.87 MJ/kg**.

### IV.C. Simulation testbed

4-DOF tool @ **1 Hz**: 3 translational DOF + 1 rotational DOF (bank angle $\mu$). Only gravity and aerodynamic accelerations modeled (RCS neglected). Bank angle tracks the guidance command with a **rate limit of 15 deg/s**. Vehicle assumed statically trimmed, constant mass.

### IV.D. Monte Carlo dispersions

| Parameter | Dispersion | Mean | 3σ or Min/Max |
|---|---|---|---|
| EI longitude | Gaussian | 0 deg | ±1.5 deg |
| EI latitude | Gaussian | 0 deg | ±0.15 deg |
| EI velocity | Gaussian | 5687 m/s | ±4.5 m/s |
| EI flight path angle | Gaussian | −10.24 deg | ±0.6 deg |
| EI azimuth | Gaussian | 38.04 deg | ±0.15 deg |
| Atmospheric density (δρ/ρ) | Uniform | 0% | ±50% |
| Vehicle mass | Uniform | 1089 kg | ±1% |
| Drag coefficient | Uniform | f(Mach,α) | ±5% |
| Lift coefficient | Uniform | f(Mach,α) | ±10% |

### IV.E. Performance criteria

Performance = offset to targeted orbit, or the **$\Delta V$ correction cost** to reach the desired orbit (sum of elementary $\Delta V_i$ for apoapsis, periapsis, inclination). At exit the periapsis is always inside the atmosphere and must be raised → the **minimum overall correction cost is the nominal periapsis correction = 113 m/s**. Bank-angle consumption is also compared as a proxy for RCS usage.

### IV.F. Classic baseline: Feedback Trajectory Control (FTC)

FTC is built on a **virtual reference trajectory based only on apoapsis control**, on which it performs a PID-like enslavement. It **decouples in-plane (apoapsis) and out-of-plane (inclination) motion**.

- **In-plane** = extension of the **Cerimele-Gamble** scheme to the whole atmospheric path. The reference trajectory (a constant-bank profile reaching the targeted apoapsis without inclination control) gives the evolution of $\cos\mu_{ref}$, $\dot h_{ref}$, $q_{ref}$ vs orbital energy. Commanded bank via its cosine:

$$\cos\mu_{com} = \cos\mu_{ref} + G_{\dot h}\,\frac{\dot h - \dot h_{ref}}{q} + G_{q}\,\frac{q - q_{ref}}{q} \tag{10}$$

If $\lvert\cos\mu_{com}\rvert > 1$, the commanded bank is set to 0° or 180° according to the sign. *(This is the exact ancestor of the repo's FTC `cos_bank` clamp + `securize_cos_bank`.)*

- **Out-of-plane** = the **roll-reversal** technique (Cerimele-Gamble logic): a roll reversal is triggered each time the inclination offset overshoots a predefined inclination corridor defined w.r.t. velocity. Roll-rate saturation enforces achievability between guidance calls.

**FTC performance (1000 MC):**

| Parameter | Mean | Std | Max | Min |
|---|---|---|---|---|
| Max g load (g) | 2.63 | 0.06 | 3.14 | 2.57 |
| Max heat flux (kW/m²) | 701.88 | 15.44 | 766.52 | 652.44 |
| Bank angle consumption (deg) | 1879.21 | 472.69 | 3348.79 | 823.92 |
| $Z_a$ offset (km) | 53.67 | 12.78 | 67.46 | 19.09 |
| $Z_p$ offset (km) | −53.06 | 13.70 | −13.94 | −68.25 |
| $i$ offset (deg) | 0.08 | 0.13 | 0.40 | −0.31 |
| Correction cost $\Delta V$ (m/s) | 144.81 | 7.00 | 159.21 | 123.33 |

Inclination corridor tuned to trigger **3–5 roll reversals** in dispersed conditions. Heat-flux and g-load requirements (1 MW/m² and 10 g for the robotic mission) met with large margins.

### IV.G. Neural aerocapture scheme

**One** feed-forward net, one hidden layer: **5 inputs → 12 hidden → 2 outputs**.
Input vector: **apoapsis (via orbital energy), eccentricity, inclination of the current orbit, current velocity, non-gravitational acceleration.**
Bank angle decoded from the 2-vector output:

$$\sin\mu_{com} = \frac{\text{output}(1)}{\lVert\text{output}\rVert}, \qquad \cos\mu_{com} = \frac{\text{output}(2)}{\lVert\text{output}\rVert} \tag{11}$$

> **This `(sin, cos)` → `atan2` decoding is exactly the repo's `atan2_signed` output parameterization.** The new article can cite Eq. (11) as the origin and contrast it with the newer `scaled_pi` / `delta` / `acos_tanh` decoders that attack the ±π wrap seam.

**Training:** 500 generations, two populations of 20, mutation rate 0.01, **600 dispersed sims** (disjoint from the 1000 MC comparison cases). Cost function strongly favors no-crash / no-hyperbolic; once satisfied, it decreases with correction cost and bank-angle consumption.

**Neural performance (1000 MC):**

| Parameter | Mean | Std | Max | Min |
|---|---|---|---|---|
| Max g load (g) | 2.28 | 0.18 | 3.08 | 1.89 |
| Max heat flux (kW/m²) | 664.68 | 29.8 | 763.92 | 565.76 |
| Bank angle consumption (deg) | 1716.17 | 441.55 | 3160.46 | 392.27 |
| $Z_a$ offset (km) | −6.75 | 16.30 | 142.28 | −39.72 |
| $Z_p$ offset (km) | 2.16 | 4.76 | 24.12 | −8.40 |
| $i$ offset (deg) | −0.03 | 0.00 | 0.02 | −0.04 |
| Correction cost $\Delta V$ (m/s) | 116.74 | 2.97 | 138.27 | 113.05 |

### IV.H. Synthesis — FTC vs Neural (the headline table)

| Performance | FTC mean | FTC worst | NN mean | NN worst |
|---|---|---|---|---|
| Max g load (g) | 2.63 | 3.14 | 2.28 (−13%) | 3.08 (−2%) |
| Max heat flux (kW/m²) | 701.88 | 766.52 | 664.68 (−5%) | 763.92 (−1%) |
| Bank angle consumption (deg) | 1879.21 | 3348.79 | 1716.17 (−9%) | 3160.46 (−6%) |
| \|$Z_a$\| offset (km) | 53.67 | 67.46 | 6.75 (−87%) | 142.28 (+110%) |
| \|$Z_p$\| offset (km) | 53.06 | 68.25 | 2.16 (−96%) | 24.12 (−65%) |
| \|$i$\| offset (deg) | 0.08 | 0.40 | 0.03 (−63%) | 0.04 (−90%) |
| Correction cost $\Delta V$ (m/s) | 144.81 | 159.21 | 116.74 (−19%) | 138.27 (−13%) |

> Mean $\Delta V$ drops **144.8 → 116.7 m/s (−19%)**, approaching the 113 m/s floor. Why: "the training of the neural guidance deals with **both in-plane and out-of-plane logics as a whole** and not separately as it is done with the FTC guidance scheme." The one $Z_a$ worst-case regression (+110%) is the honest caveat worth keeping.

---

## V. Conclusion — and the explicit hook for the new article

- GA-trained NNs successfully perform both soft-landing (6-DOF) and aerocapture (4-DOF) guidance.
- Soft-landing: higher accuracy + behavior tunable through the fitness function (lower consumption *or* gentler maneuvering).
- Aerocapture: simultaneous in-plane/out-of-plane handling, higher exit accuracy → lower correction cost, lower bank-angle consumption.
- **Drawback:** "the heavy computational burden of its training which cannot be performed on-board... drastic changes in the mission parameters are not possible during the flight."
- **Stated next step (verbatim):** *"The next step would be to extend our work on the aerocapture to skip entry missions and evaluate the performance of neural guidance compared to classic algorithms such as the predictor-corrector schemes."*

> **This sentence is the seed of the new paper.** The repo already implements predictor-corrector baselines (FNPAG = Lu's numerical predictor-corrector, PredGuid = Apollo/Shuttle drag tracking) — exactly the comparison the 2009 conclusion called for, 17 years later.

---

## References (2009 paper)

1. Pignie G., *Navigation, Guidance and Control of the Atmospheric Reentry Demonstrator Preliminary Flight Results*, AAAF, 1999.
2. Jategaonkar R., Behr R., Gockel W., Zorn C., *Data Analysis of Phoenix RLV Demonstrator Flight Test*, AIAA 2005-6129.
3. Vernis P., Gelly G., Ferreira E., da Costa R., Ortega G., *Guidance Trade-Off for Aerocapture Missions*, ESA GNC, 2004.
4. Strandmoe S., Jean-Marius T., Trinh S., *Vision Based Autonomous Planetary Lander*, AIAA-1999-4154.
5. Frapard B., Champetier C., Kemble S., Parkinson B., Strandmoe S., Lang M., *Vision-Based GNC Design for the LEDA Mission*, ESA GNC, 1996.
6. Anderson D., McNeill G., *Artificial Neural Networks Technology*, ADACS State-of-the-Art Report, 1992.
7. Zurada J. M., *Introduction to Artificial Neural Systems*, West Publishing Company, 1992.
8. Werbos P., *Backpropagation through time: what it does and how to do it*, Proceedings of the IEEE, pp. 1150–1560, 1990.
9. Ferrari S., Stengel R. F., *Online adaptive critic flight control*, Journal of Guidance, Control and Dynamics, vol. 27, no. 5, Sept–Oct 2004.
10. Cherry G. W., *A general explicit, optimizing guidance law for rocket-propelled spacecraft*, Apollo Guidance and Navigation, 1964.
11. Gelly G., Ferreira E., *Guidance Algorithm Using Neural Networks for a Soft-Landing Application*, EUCASS, 2007.
12. Cerimele C. J., Gamble J. D., *A Simplified Guidance Algorithm for Lifting Aeroassist Orbital Transfer Vehicles*, AIAA-85-0348.
