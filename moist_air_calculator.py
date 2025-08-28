# Moist Air Calculator — Streamlit
# Text-input numeric fields so you can highlight/backspace/paste, then press Enter to apply.
# Features:
#  • Inputs: DB + (RH or WB), pressure (sea level or custom)
#  • Outputs: named psychrometrics + transport properties
#  • Water content (g/kg_da, g/s) and condensate when cooling to saturation
#  • Robust parsing (accepts commas or scientific notation), clear validation

import sys, platform
import streamlit as st
import numpy as np
from scipy.optimize import root_scalar
from CoolProp.CoolProp import HAPropsSI

st.set_page_config(page_title="Moist Air Calculator", layout="wide")
ATM_P = 101325.0

# -------------------- helpers: numeric text inputs --------------------
def parse_number(text, *, min_val=None, max_val=None, field_name="value"):
    """
    Parse a numeric string (accepts commas, sci-notation). Returns float.
    Raises ValueError with a helpful message if invalid / out of range.
    """
    if text is None:
        raise ValueError(f"{field_name}: empty.")
    s = text.strip().replace(",", ".")
    if s == "":
        raise ValueError(f"{field_name}: empty.")
    try:
        x = float(s)
    except Exception:
        raise ValueError(f"{field_name}: '{text}' is not a number.")
    if min_val is not None and x < min_val:
        raise ValueError(f"{field_name}: {x} < min {min_val}.")
    if max_val is not None and x > max_val:
        raise ValueError(f"{field_name}: {x} > max {max_val}.")
    return x

def text_num(label, key, default_str, help=None):
    # initialize session default
    if key not in st.session_state:
        st.session_state[key] = default_str
    return st.text_input(label, key=key, help=help)

# -------------------- humid-air property functions --------------------
def humid_air_props(T_K, P_Pa=ATM_P, RH=None, W=None):
    """
    Return humid-air properties at (T, P) using either RH or humidity ratio W.
    Keys:
      T[K], RH[-], W[kg/kg_da], h[J/kg_da], Tdp[K], Twb[K],
      rho[kg/m3], mu[Pa·s], k[W/m·K], cp[J/kg·K], Pr[-], nu[m2/s], alpha[m2/s]
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

    # psychrometrics
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

    # derived
    Pr = cp*mu/max(k, 1e-12)
    nu = mu/max(rho, 1e-12)
    alpha = k/max(rho*cp, 1e-12)

    return {'T':T_K,'RH':RH,'W':W,'h':h,'Tdp':Tdp,'Twb':Twb,
            'rho':rho,'mu':mu,'k':k,'cp':cp,'Pr':Pr,'nu':nu,'alpha':alpha}

def state_from_DB_RH(Tdb_C, RH_pct, P=ATM_P):
    T = Tdb_C + 273.15
    RH = float(np.clip(RH_pct/100.0, 1e-4, 0.999))
    s = humid_air_props(T, P, RH=RH); s['T'] = T; return s

def state_from_DB_WB(Tdb_C, Twb_C, P=ATM_P):
    T = Tdb_C + 273.15
    Twb = min(Twb_C, Tdb_C) + 273.15
    def f(W): return HAPropsSI('Twb','T',T,'P',P,'W',W) - Twb
    W_lo = 1e-7
    try:
        W_hi = HAPropsSI('W','T',T,'P',P,'R',0.999)
    except Exception:
        W_hi = 0.03
    try:
        sol = root_scalar(f, bracket=[W_lo, W_hi], method='bisect', xtol=1e-7)
        W = sol.root
    except Exception:
        W = HAPropsSI('W','T',T,'P',P,'R',0.5)
    s = humid_air_props(T, P, W=W); s['T'] = T; return s

def final_state_after_Qdot(s1, Qdot_kW, m_dot_air, P=ATM_P):
    """Return (s2, note, condensate_dict|None)."""
    if abs(Qdot_kW) < 1e-12 or m_dot_air <= 1e-12:
        return s1.copy(), "No change (zero heat or zero mass flow).", None

    h1 = s1['h']; W1 = s1['W']; T1 = s1['T']
    q_per_kg = (Qdot_kW*1000.0)/m_dot_air  # J per kg dry air
    h2_target = h1 + q_per_kg

    # Dry (constant-W) attempt
    def h_at_T_constW(T): return HAPropsSI('H','T',T,'P',P,'W',W1)
    T_lo = max(173.15, T1 - 100.0); T_hi = min(373.15, T1 + 100.0)
    try:
        T2_dry = root_scalar(lambda T: h_at_T_constW(T) - h2_target,
                             bracket=[T_lo, T_hi], method='bisect').root
        RH2 = HAPropsSI('R','T',T2_dry,'P',P,'W',W1)
        if RH2 <= 0.999 or q_per_kg >= 0:
            s2 = humid_air_props(T2_dry, P, W=W1); s2['T'] = T2_dry
            return s2, "Dry process (W constant).", None
    except Exception:
        pass

    # Saturated (condensation) solution
    def h_sat(T):
        Wsat = HAPropsSI('W','T',T,'P',P,'R',0.999)
        return HAPropsSI('H','T',T,'P',P,'W',Wsat)
    try:
        T2 = root_scalar(lambda T: h_sat(T) - h2_target,
                         bracket=[max(173.15, T1-80.0), T1], method='bisect').root
        W2 = HAPropsSI('W','T',T2,'P',P,'R',0.999)
        s2 = humid_air_props(T2, P, W=W2); s2['T'] = T2

        dW = max(0.0, W1 - W2)                # kg/kg_da
        mdot_cond_kg_s = dW * m_dot_air       # kg/s
        condensate = {
            'dW_g_per_kg': 1000.0*dW,
            'mdot_g_s': 1000.0*mdot_cond_kg_s,
            'mdot_kg_h': 3600.0*mdot_cond_kg_s,
            'vol_mL_s': 1000.0*mdot_cond_kg_s,  # ≈ 1 g/mL
            'vol_L_h': 3.6*mdot_cond_kg_s
        }
        return s2, "Cooling with condensation (final state saturated).", condensate
    except Exception as e:
        # Fallback: constant-W estimate
        T2_guess = np.clip(T1 + q_per_kg/max(s1['cp'],1e-9), T_lo, T_hi)
        s2 = humid_air_props(T2_guess, P, W=W1); s2['T'] = T2_guess
        return s2, f"Dry fallback — check inputs. ({e})", None

# -------------------- UI --------------------
st.title("Moist Air Calculator (free-typing inputs + Enter)")

with st.sidebar.form("inputs_form", clear_on_submit=False):
    st.header("Inputs")

    P_mode = st.selectbox("Pressure mode", ["Sea level (101325 Pa)", "Custom (Pa)"], index=0)
    P_txt  = text_num("Pressure (Pa)", "txt_P", "101325", help="Only used if 'Custom' is selected.")

    mode = st.radio("Moisture input mode", ["DB + RH", "DB + WB"], index=0)
    Tdb_txt = text_num("Dry-bulb (°C)", "txt_Tdb", "30.0")

    if mode == "DB + RH":
        RH_txt  = text_num("Relative Humidity (%)", "txt_RH", "50.0")
        Twb_txt = None
    else:
        Twb_txt = text_num("Wet-bulb (°C)", "txt_Twb", "20.0")
        RH_txt  = None

    st.header("Process (Q̇ on flowing air)")
    mdot_txt = text_num("Air mass flow ṁ_air (kg/s, dry air basis)", "txt_mdot", "1.0")
    qdot_txt = text_num("Heat rate Q̇ (kW)  (+heating / −cooling)", "txt_qdot", "-5.0")

    submitted = st.form_submit_button("Update / Calculate", use_container_width=True)

# ---- parse & validate all inputs (after submit or initial render) ----
errors = []

# Pressure
try:
    P_val = parse_number(P_txt, min_val=50000, max_val=120000, field_name="Pressure")
    P = P_val if P_mode.startswith("Custom") else ATM_P
except ValueError as e:
    errors.append(str(e))
    P = ATM_P

# DB
try:
    Tdb_C = parse_number(Tdb_txt, min_val=-60.0, max_val=120.0, field_name="Dry-bulb")
except ValueError as e:
    errors.append(str(e))
    Tdb_C = 30.0

# RH/WB
RH_pct = None; Twb_C = None
if mode == "DB + RH":
    try:
        RH_pct = parse_number(RH_txt, min_val=1.0, max_val=99.0, field_name="Relative Humidity (%)")
    except ValueError as e:
        errors.append(str(e))
        RH_pct = 50.0
else:
    try:
        Twb_C = parse_number(Twb_txt, min_val=-60.0, max_val=120.0, field_name="Wet-bulb")
    except ValueError as e:
        errors.append(str(e))
        Twb_C = 20.0
    if Twb_C > Tdb_C:
        errors.append("Wet-bulb cannot exceed Dry-bulb. It will be clamped to DB.")
        Twb_C = Tdb_C

# process inputs
try:
    m_dot_air = parse_number(mdot_txt, min_val=0.0, max_val=500.0, field_name="Air mass flow")
except ValueError as e:
    errors.append(str(e))
    m_dot_air = 1.0

try:
    Qdot_kW = parse_number(qdot_txt, min_val=-10000.0, max_val=10000.0, field_name="Heat rate")
except ValueError as e:
    errors.append(str(e))
    Qdot_kW = -5.0

# Show all validation errors prominently (but still compute with safe defaults)
if errors:
    st.error("Please fix these inputs:")
    for e in errors:
        st.write("• " + e)

# ---- compute states ----
s1 = state_from_DB_RH(Tdb_C, RH_pct, P) if RH_pct is not None else state_from_DB_WB(Tdb_C, Twb_C, P)
s2, note, condensate = final_state_after_Qdot(s1, Qdot_kW, m_dot_air, P)

# -------------------- display --------------------
def named_block(label, items):
    st.markdown(f"### {label}")
    for name, value, unit in items:
        st.write(f"**{name}**: {value} {unit}")
    st.divider()

def state_blocks(title, s, m_dot_air=None):
    T_C   = s['T'] - 273.15
    Twb_C = s['Twb'] - 273.15
    Tdp_C = s['Tdp'] - 273.15
    v     = 1.0/max(s['rho'],1e-12)
    W_gkg = s['W']*1000.0
    items_main = [
        ("Dry-bulb temperature (DB)", f"{T_C:.2f}", "°C"),
        ("Wet-bulb temperature (WB)", f"{Twb_C:.2f}", "°C"),
        ("Dew-point temperature (DP)", f"{Tdp_C:.2f}", "°C"),
        ("Relative humidity (RH)", f"{s['RH']*100:.2f}", "%"),
        ("Humidity ratio (W)", f"{W_gkg:.3f}", "g/kg dry air"),
        ("Enthalpy (h)", f"{s['h']/1000.0:.3f}", "kJ/kg dry air"),
    ]
    items_props = [
        ("Density (ρ)", f"{s['rho']:.4f}", "kg/m³"),
        ("Specific volume (v)", f"{v:.4f}", "m³/kg dry air"),
        ("Specific heat (cp)", f"{s['cp']:.1f}", "J/kg·K"),
        ("Thermal conductivity (k)", f"{s['k']:.5f}", "W/m·K"),
        ("Dynamic viscosity (μ)", f"{s['mu']:.7f}", "Pa·s"),
        ("Prandtl number (Pr)", f"{s['Pr']:.3f}", "–"),
        ("Kinematic viscosity (ν)", f"{s['nu']:.7e}", "m²/s"),
        ("Thermal diffusivity (α)", f"{s['alpha']:.7e}", "m²/s"),
    ]

    st.subheader(title)
    named_block("Psychrometric state", items_main)
    named_block("Transport & derived properties", items_props)

    if m_dot_air and m_dot_air > 0:
        water_g_s = s['W']*m_dot_air*1000.0
        named_block("Water content in the stream", [("Water mass flow (ṁ_w)", f"{water_g_s:.2f}", "g/s")])

col1, col2 = st.columns(2)
with col1:
    state_blocks("Initial State", s1, m_dot_air=m_dot_air)
with col2:
    state_blocks("Final State (after Q̇)", s2, m_dot_air=m_dot_air)
    st.info(note)
    if condensate is not None:
        st.markdown("### Condensate removed (due to cooling)")
        for name, value, unit in [
            ("Water removed per kg dry air (ΔW)", f"{condensate['dW_g_per_kg']:.2f}", "g/kg dry air"),
            ("Condensate mass flow", f"{condensate['mdot_g_s']:.2f}", "g/s"),
            ("Condensate mass flow", f"{condensate['mdot_kg_h']:.3f}", "kg/h"),
            ("Condensate volume flow (≈ water)", f"{condensate['vol_mL_s']:.1f}", "mL/s"),
            ("Condensate volume flow (≈ water)", f"{condensate['vol_L_h']:.3f}", "L/h"),
        ]:
            st.write(f"**{name}**: {value} {unit}")

with st.expander("App diagnostics", expanded=False):
    st.write({"CoolProp": getattr(sys.modules.get('CoolProp'), '__version__', 'unknown'),
              "Python": sys.version.split()[0],
              "Platform": platform.platform()})

st.caption("Type freely in any field (commas or scientific notation OK), then press Enter or click the button to apply.")
