# Physics background

## 1. Spacetime metric

pyRNS models stationary, axisymmetric neutron stars in general relativity.
The metric is written in **quasi-isotropic coordinates** (Komatsu et al. 1989):

$$
ds^2 = -e^{2\nu}\,dt^2
       + e^{2\zeta}\!\left(dr_{\rm is}^2 + r_{\rm is}^2\,d\theta^2\right)
       + e^{2\psi}\,r_{\rm is}^2\sin^2\!\theta\,(d\phi - \omega\,dt)^2
$$

where all four **metric potentials** (ν, ζ, ψ, ω) depend only on the
isotropic radial coordinate $r_{\rm is}$ and polar angle θ.

### Internal code potentials

The code stores four combinations that simplify the field equations:

| Symbol | Definition | Physical meaning |
|--------|-----------|-----------------|
| `gama` | $\nu + \zeta$ | lapse + conformal factor |
| `rho`  | $\nu - \zeta$ | lapse − conformal factor |
| `alpha`| $\zeta$ | spatial conformal factor |
| `omega`| $\omega$ | frame-dragging angular velocity |

The lapse is recovered as $e^\nu = e^{(\texttt{gama}+\texttt{rho})/2}$ and
the spatial conformal factor as $e^\zeta = e^{(\texttt{gama}-\texttt{rho})/2}$.

---

## 2. Compactified radial coordinate

All grid functions are evaluated on the compact domain

$$
s = \frac{r_{\rm is}}{r_{\rm is} + r_e}, \qquad s \in [0,\, s_{\rm max})
$$

where $r_e$ is the isotropic equatorial radius of the star.
This maps the half-line $r_{\rm is}\in[0,\infty)$ to a finite interval,
allowing the exterior (vacuum) to be included in the grid without truncation.

The grid has `SDIV = 129` radial and `MDIV = 65` angular points:

$$
s_j = s_{\rm max}\,\frac{j-1}{N_s - 1}, \qquad
\mu_m = \frac{m-1}{N_\mu - 1}
$$

where $\mu = \cos\theta$ runs from 0 (equator) to 1 (pole).

---

## 3. Tolman–Oppenheimer–Volkoff equation

For the **non-rotating** (spherically symmetric) initial configuration,
pyRNS integrates the TOV equations in isotropic coordinates:

$$
\frac{dp}{dr_{\rm is}} = -\frac{(\varepsilon+p)\,(m + 4\pi r^3 p)}{r\,r_{\rm is}\,\sqrt{1-2m/r}}
$$

$$
\frac{dm}{dr_{\rm is}} = 4\pi\,\varepsilon\,\frac{r^3}{r_{\rm is}}\,\sqrt{1-2m/r}
$$

$$
\frac{dr}{dr_{\rm is}} = \frac{r}{r_{\rm is}}\,\sqrt{1-2m/r}
$$

where $r$ is the Schwarzschild radial coordinate, related to the isotropic
coordinate $r_{\rm is}$ by the conformal factor
$r = r_{\rm is}\,e^{\zeta}$.

Integration proceeds outward from the centre (with Taylor-series initial
conditions) until the pressure drops to the surface value $p_{\rm surf}$.

### Metric functions from TOV

After integration the metric potentials are set from:

$$
\nu_{\rm surf} = \ln\!\left(\frac{1 - m_f/(2 r_e)}{1 + m_f/(2 r_e)}\right), \qquad
\zeta = \ln\!\left(\frac{r}{r_{\rm is}}\right)
$$

where $m_f$ is the total gravitational mass and $r_e$ the isotropic
surface radius.

---

## 4. Hydrostatic equilibrium (rotating case)

For a **uniformly rotating** star the Bernoulli integral gives the specific
enthalpy at each grid point:

$$
h(r_{\rm is},\theta) = h_{\rm pole} + \tfrac{1}{2}\,r_e^2\,\bigl[
  (\gamma+\rho)_{\rm pole} - (\gamma+\rho)(r_{\rm is},\theta)
\bigr] - \tfrac{1}{2}\ln(1 - v^2)
$$

where $h_{\rm pole}$ is the enthalpy at the polar surface (= 0),
and the fluid 3-velocity relative to the ZAMO frame is

$$
v = (\Omega - \omega)\,\frac{r_{\rm is}\sin\theta}{e^{(\gamma+\rho)/2}}
$$

Matter exists where $h > h_{\rm min}$ (a small threshold that
sets the stellar surface).

---

## 5. Self-consistent field iteration (KEH method)

The rotating-star metric is found by iterating:

1. **Guess** metric potentials (from the non-rotating sphere).
2. **Compute** the matter distribution via the Bernoulli integral.
3. **Compute** the source terms $S_\rho$, $S_\gamma$, $S_\omega$ from
   the stress-energy tensor and metric coupling.
4. **Update** the metric by solving the radial ODEs:
   $$\nabla^2_{\rm flat}\,u_n = S_n(r_{\rm is})$$
   for each Legendre mode $n = 0,\ldots,N_{\rm max}$ (`LMAX = 10`).
5. **Integrate** the constraint equation for $\alpha = \zeta$ along the
   μ-direction.
6. **Repeat** until $|r_e^{\rm new} - r_e^{\rm old}|/r_e < \epsilon$
   (default $\epsilon = 10^{-5}$).

> **Note:** The rotating-star iteration (step 4) requires an exact
> normalisation of the Green's-function kernel that is only available in
> the original Stergioulas C source. The current Python implementation
> converges correctly for non-rotating stars ($r_{\rm ratio}=1$) but
> diverges for $r_{\rm ratio}<1$.

---

## 6. Global quantities

### Gravitational mass

$$
M = 4\pi\,r_e^3 \int_0^1\!\int_0^1
  e^{2\alpha+\gamma}\left[\frac{(\varepsilon+p)(1+v^2)}{1-v^2} + 2p\right]
  \frac{s^2}{(1-s)^4}\,d\mu\,ds
$$

(tabulated EOS: multiply by $\sqrt{\kappa}\,c^2/G$ to get grams.)

### Baryon mass

$$
M_0 = 4\pi\,r_e^3 \int_0^1\!\int_0^1
  e^{2\alpha+(\gamma-\rho)/2}\,\frac{\rho_0}{\sqrt{1-v^2}}
  \frac{s^2}{(1-s)^4}\,d\mu\,ds
$$

### Angular momentum

$$
J = 4\pi\,r_e^4 \int_0^1\!\int_0^1
  e^{2\alpha+\gamma-\rho}\,\sin\theta\,
  \frac{(\varepsilon+p)\sqrt{v^2}}{1-v^2}
  \frac{s^2}{(1-s)^4}\,d\mu\,ds
$$

### Circumferential equatorial radius

$$
R_e = r_e\,e^{\zeta_{\rm eq}} = r_e\,e^{(\gamma-\rho)_{\rm eq}/2}
$$

For the tabulated EOS multiply by $\sqrt{\kappa}$ to obtain centimetres.

### Keplerian angular velocity

Derived from the Schwarzschild approximation at the stellar surface:

$$
\Omega_K = \sqrt{\frac{M_{\rm code}}{R_{S,\,\rm code}^3}}\,,\quad
\Omega_K^{\rm phys} = \Omega_K^{\rm code}\,\frac{c}{\sqrt{\kappa}}
$$

where $R_{S,\rm code}$ is the Schwarzschild surface radius in code units.

---

## 7. Unit systems

### Code units (polytrope)

For the polytropic EOS (p = ρ₀^Γ) there is no intrinsic physical scale
unless the polytropic constant $K$ is specified.  All results are
dimensionless; the natural characterisation is the compactness $M/R_e$
and the dimensionless Keplerian frequency $\Omega_K\sqrt{R_e^3/M}$ (both
in geometric units G = c = 1).

### Code units (tabulated EOS)

The tabulated solver uses the length scale

$$
\sqrt{\kappa} = \sqrt{\frac{c^2}{G}\times 10^{-15}}\approx 3.67\times 10^6\ \text{cm}^{1/2}
$$

Conversions:

| Quantity | Code → physical |
|----------|----------------|
| Length $r_{\rm code}$ | $r_{\rm phys}\ [\text{cm}] = r_{\rm code}\,\sqrt{\kappa}$ |
| Mass $M_{\rm code}$ | $M_{\rm phys}\ [\text{g}] = M_{\rm code}$ (already grams from the integral) |
| Energy density $\varepsilon_{\rm code}$ | $\varepsilon_{\rm phys}\ [\text{g\,cm}^{-3}] = \varepsilon_{\rm code}/(c^2\,\kappa_{\rm scale})$ |
| Pressure $p_{\rm code}$ | $p_{\rm phys}\ [\text{dyne\,cm}^{-2}] = p_{\rm code}/\kappa_{\rm scale}$ |
| Ang. vel. $\Omega_{\rm code}$ | $\Omega_{\rm phys}\ [\text{rad\,s}^{-1}] = \Omega_{\rm code}\,c/\sqrt{\kappa}$ |

where $\kappa_{\rm scale} = \kappa G/c^4 \approx 1.11\times10^{-35}$.

---

## References

1. **Komatsu, Eriguchi & Hachisu** (1989), *MNRAS* **237**, 355 —
   original KEH self-consistent field method.
2. **Cook, Shapiro & Teukolsky** (1994), *ApJ* **422**, 227 —
   improved KEH with compactified coordinates.
3. **Stergioulas & Friedman** (1995), *ApJ* **444**, 306 —
   RNS code description and tests.
4. **Friedman & Stergioulas** (2013), *Rotating Relativistic Stars*,
   Cambridge University Press — comprehensive textbook treatment.
