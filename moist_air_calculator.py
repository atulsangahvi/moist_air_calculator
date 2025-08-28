# Moist Air Calculator — Streamlit
# - Inputs: DB + (RH or WB), pressure
# - Outputs: full moist-air properties (Tdb, Twb, RH, W, h, Tdp, rho, v, cp, k, mu, Pr)
# - Process: apply Q̇ (kW) to airstream with ṁ_air (kg/s). Handles:
#     • Heating/cooling with constant W (dry)
#     • If cooling drives RH>100%, solve with condensation (saturated final state)

import streamlit as st
import numpy as np
from scipy.optimize import root_scalar
from CoolProp.CoolProp import HAPropsSI

ATM_P = 101325.0

# ---------- Robust humid-air helpers ----------
def humid_air_props(T_K, P_Pa=ATM_P, RH=None, W=None):
    """
    Return humid-air properties at (T, P) using either RH or humidity ratio W.
    Outputs: dict with keys:
      'rho' [kg/m3], 'mu' [Pa·s], 'k' [W/m·K], 'cp' [J/kg·K],
      'Pr' [-], 'nu' [m2/s], 'alpha' [m2/s], 'W' [-], 'RH' [-],
      'h' [J/kg_da], 'Tdp' [K], 'Twb' [K]
    """
    if RH is None and W is None:
        raise ValueError("Provide either RH or W")

    # sanitize moisture input
    if RH is not None:
        RH = float(np.clip(RH, 1e-4, 0.999))
        try:
            W = HAPropsSI('W','T',T_K,'P',P_Pa,'R',RH)
        except Exception:
            RH = min(0.995, RH*0.98)
            W = HAPropsSI('W','T',T_K,'P',P_Pa,'R',RH)
    else:
        try:
            Wsat = HAPropsSI('W','T',T_K,'P',P_Pa,'R',0.999)
            W = float(min(float(W), 0.999*Wsat))
        except Exception:
            W = float(W)
        RH = HAPropsSI('R','T',T_K,'P',P_Pa,'W',W)
        RH = float(np.clip(RH, 1e-4, 0.999))

    # Primary properties
    # density
    try:
        rho = HAPropsSI('Rho','T',T_K,'P',P_Pa,'W',W)
    except Exception:
        R_d = 287.042
        rho = P_Pa/(R_d*T_K*(1.0 + 1.6078*W))
    # μ, k, cp
    try:
        mu  = HAPropsSI('M','T',T_K,'P',P_Pa,'W',W)  # dynamic viscosity
    except Exception:
        mu = 1.716e-5 * (T_K/273.15)**1.5 * (273.15+111.0)/(T_K+111.0)
    try:
        k   = HAPropsSI('K','T',T_K,'P',P_Pa,'W',W)
    except Exception:
        k = 0.024 + 7.0e-5*(T_K - 273.15)
    try:
        cp  = HAPropsSI('C','T',T_K,'P',P_Pa,'W',W)
    except Exception:
        yv = W/(1.0+W)
        cp = (1.0-yv)*1006.0 + yv*1860.0

    # Psychrometrics
    try:
        h   = HAPropsSI('H','T',T_K,'P',P_Pa,'W',W)
    except Exception:
        T_C = T_K - 273.15
        h_kJ = 1.006*T_C + W*(2501.0 + 1.86*T_C)
        h = h_kJ*1000.0
    try:
        Tdp = HAPropsSI('Tdp','T',T_K,'P',P_Pa,'W',W)
    except Exception:
        Tdp = T_K - 10.0
    try:
        Twb = HAPropsSI('Twb','T',T_K,'P',P_Pa,'W',W)
    except Exception:
        Twb = T_K - 5.0

    # secondary groups
    Pr = cp*mu/max(k, 1e-12)
    nu = mu/max(rho, 1e-12)
    alpha = k/max(rho*cp, 1e-12)

    return {'rho':rho, 'mu':mu, 'k':k, 'cp':cp,
            'Pr':Pr, 'nu':nu, 'alpha':alpha, 'W':W, 'RH':RH,
            'h':h, 'Tdp':Tdp, 'Twb':Twb}

def state_from_DB_RH(Tdb_C, RH_pct, P=ATM_P):
    T = Tdb_C + 273.15
    RH = float(np.clip(RH_pct/100.0, 1e-4, 0.999))
    p = humid_air_props(T, P, RH=RH)
    p['T'] = T
    return p

def state_from_DB_WB(Tdb_C, Twb_C, P=ATM_P):
    if Twb_C > Tdb_C:
        Twb_C = Tdb_C
    T = Tdb_C + 273.15
    Twb = Twb_C + 273.15

    # invert Twb(T, W) -> W at given DB
    def f(W):
        return HAPropsSI('Twb','T',T,'P',P,'W',W) - Twb

    # bracket W between very dry and near saturation
    W_lo = 1e-7
    try:
        W_hi = HAPropsSI('W','T',T,'P',P,'R',0.999)
    except Exception:
        W_hi = 0.03

    try:
        sol = root_scalar(f, bracket=[W_lo, W_hi], method='bisect', xtol=1e-7)
        W = sol.root
    except Exception:
        # fallback: use 50% RH
        W = HAPropsSI('W','T',T,'P',P,'R',0.5)

    p = humid_air_props(T, P, W=W)
    p['T'] = T
    return p

def display_state(label, s):
    T_C   = s['T'] - 273.15
    Twb_C = s['Twb'] - 273.15
    Tdp_C = s['Tdp'] - 273.15
    v     = 1.0/max(s['rho'],1e-12)
    st.markdown(f"**{label}**")
    st.write(f"DB = {T_C:.2f} °C | WB = {Twb_C:.2f} °C | DP = {Tdp_C:.2f} °C | RH = {s['RH']*100:.2f} %")
    st.write(f"W = {s['W']*1000:.3f} g/kg_da | h = {s['h']/1000:.3f} kJ/kg_da | ρ = {s['rho']:.3f} kg/m³ | v = {v:.3f} m³/kg_da")
    st.write(f"cp = {s['cp']:.1f} J/kg·K | k = {s['k']:.4f} W/m·K | μ = {s['mu']:.7f} Pa·s | Pr = {s['Pr']:.3f}")
    st.divider()

# ---------- Solve resulting state after Qdot ----------
def final_state_after_Qdot(s1, Qdot_kW, m_dot_air, P=ATM_P):
    """
    Given initial state dict s1, heat rate Qdot [kW] (+heating, -cooling),
    and air mass flow m_dot_air [kg_da/s], return final state s2.
    Handles:
      - Dry heating/cooling with constant W if final RH <= 100%
      - If cooling would exceed saturation, solve saturated final state (condensation).
    """
    if abs(Qdot_kW) < 1e-12 or m_dot_air <= 1e-12:
        return s1.copy(), "No change (zero heat or zero mass flow)."

    h1 = s1['h']
    W1 = s1['W']
    T1 = s1['T']

    q_per_kg = (Qdot_kW*1000.0)/m_dot_air  # J/kg_da
    h2_target = h1 + q_per_kg

    # Dry path (constant W)
    def h_at_T_constW(T):  # enthalpy at temp T and W = W1
        return HAPropsSI('H','T',T,'P',P,'W',W1)

    # Try dry solution
    # sensible heating/cooling bracket:  -60°C..100°C around T1
    T_lo = max(173.15, T1 - 100.0)
    T_hi = min(373.15, T1 + 100.0)

    try:
        f = lambda T: h_at_T_constW(T) - h2_target
        sol = root_scalar(f, bracket=[T_lo, T_hi], method='bisect')
        T2_dry = sol.root
        # Check saturation
        RH2 = HAPropsSI('R','T',T2_dry,'P',P,'W',W1)
        if RH2 <= 0.999 or q_per_kg >= 0:
            # Accept dry solution (heating always OK; cooling OK if not supersaturated)
            s2 = humid_air_props(T2_dry, P, W=W1); s2['T'] = T2_dry
            note = "Dry process (W constant)."
            return s2, note
        # else supersaturated: fall through to saturated solution
    except Exception:
        # fall through to saturated solution if dry solve fails
        pass

    # Saturated (condensation) path: find T so that h(T, Wsat(T)) = h2_target
    def h_sat(T):
        Wsat = HAPropsSI('W','T',T,'P',P,'R',0.999)
        return HAPropsSI('H','T',T,'P',P,'W',Wsat)

    # bracket: between ~ -20°C and T1 (can't be warmer than start for net cooling w/ condensation)
    T_lo_sat = max(173.15, T1 - 80.0)
    T_hi_sat = T1
    try:
        g = lambda T: h_sat(T) - h2_target
        sol2 = root_scalar(g, bracket=[T_lo_sat, T_hi_sat], method='bisect')
        T2 = sol2.root
        W2 = HAPropsSI('W','T',T2,'P',P,'R',0.999)
        s2 = humid_air_props(T2, P, W=W2); s2['T'] = T2
        note = "Cooling with condensation (final state saturated)."
        return s2, note
    except Exception as e:
        # As a last resort, return the dry solution even if RH>100% (flag it)
        T2_guess = np.clip(T1 + q_per_kg/max(s1['cp'],1e-9), T_lo, T_hi)
        s2 = humid_air_props(T2_guess, P, W=W1); s2['T'] = T2_guess
        return s2, "Dry solution fallback — RH may exceed 100% (check inputs)."

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Moist Air Calculator", layout="wide")
st.title("Moist Air Calculator (DB + RH/WB, with Q̇ process)")

with st.sidebar:
    st.header("Inputs")
    P_mode = st.selectbox("Pressure mode", ["Sea level (101325 Pa)", "Custom (Pa)"], index=0)
    if P_mode.startswith("Custom"):
        P = st.number_input("Pressure (Pa)", min_value=50000, max_value=120000, value=101325, step=500)
    else:
        P = ATM_P

    mode = st.radio("Moisture input mode", ["DB + RH", "DB + WB"], index=0)
    Tdb_C = st.number_input("Dry-bulb (°C)", min_value=-60.0, max_value=120.0, value=30.0, step=0.5)
    if mode == "DB + RH":
        RH_pct = st.number_input("Relative Humidity (%)", min_value=1.0, max_value=99.0, value=50.0, step=0.5)
        s1 = state_from_DB_RH(Tdb_C, RH_pct, P)
    else:
        Twb_C = st.number_input("Wet-bulb (°C)", min_value=-60.0, max_value=120.0, value=20.0, step=0.5)
        if Twb_C > Tdb_C:
            st.warning("Wet-bulb cannot exceed dry-bulb; clamped.")
            Twb_C = Tdb_C
        s1 = state_from_DB_WB(Tdb_C, Twb_C, P)

    st.header("Process (Q̇ on flowing air)")
    m_dot_air = st.number_input("Air mass flow (kg/s, dry air basis)", min_value=0.0, max_value=500.0, value=1.0, step=0.1)
    Qdot_kW   = st.number_input("Heat rate Q̇ (kW)  (+heating / −cooling)", min_value=-10000.0, max_value=10000.0, value=-5.0, step=0.5)

col1, col2 = st.columns(2)

with col1:
    display_state("Initial State", s1)

with col2:
    s2, note = final_state_after_Qdot(s1, Qdot_kW, m_dot_air, P)
    display_state("Final State (after Q̇)", s2)
    st.info(note)

# Tiny tips
st.caption("Notes: h is per kg of dry air. If cooling pushes RH above 100%, the solver switches to a saturated final state (condensation).")
