# Moist Air Calculator — HVAC Volumetric Flow (m³/s) + Dual Enthalpy + Unicode-Safe PDF
# - No SciPy required (pure-Python bisection for root-finding)
# - Pin Python 3.11 (via runtime.txt) so CoolProp/Numpy wheels are available
# - Recommend fpdf2>=2.7.8 for fewer encoding edge cases

import sys, platform, tempfile
from io import BytesIO
import streamlit as st
import numpy as np
from CoolProp.CoolProp import HAPropsSI
from fpdf import FPDF

st.set_page_config(page_title="Moist Air Calculator (HVAC volumetric + PDF)", layout="wide")
ATM_P = 101325.0

# -------------------- tiny, robust bisection solver (no SciPy) --------------------
def bisect_solve(func, a, b, *, max_iter=80, tol=1e-7):
    """Find root of continuous func in [a,b] assuming f(a)*f(b) <= 0. Returns mid if fails."""
    fa = func(a); fb = func(b)
    if not np.isfinite(fa) or not np.isfinite(fb):
        return 0.5*(a+b)
    if fa == 0.0: return a
    if fb == 0.0: return b
    if fa*fb > 0:
        # Try to expand the bracket a little if possible
        for k in range(6):
            da = (b-a)*0.1*(k+1)
            fa = func(a - da); fb = func(b + da)
            if np.isfinite(fa) and np.isfinite(fb) and fa*fb <= 0:
                a, b = a - da, b + da
                break
        else:
            return 0.5*(a+b)
    for _ in range(max_iter):
        c = 0.5*(a+b)
        fc = func(c)
        if not np.isfinite(fc):  # numerical hiccup: nudge c
            c = np.nextafter(c, b)
            fc = func(c)
        if abs(fc) < tol or 0.5*(b-a) < tol:
            return c
        if fa*fc <= 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5*(a+b)

# -------------------- numeric input helpers --------------------
def parse_number(text, *, min_val=None, max_val=None, field_name="value"):
    if text is None: raise ValueError(f"{field_name}: empty.")
    s = text.strip().replace(",", ".")
    if s == "": raise ValueError(f"{field_name}: empty.")
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

    # moisture input
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
        mu  = HAPropsSI('M','T',T_K,'P',P_Pa,'W',W)
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
        h   = HAPropsSI('H','T',T_K,'P',P_Pa,'W',W)      # J/kg_dry
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
    s = humid_air_props(T, P, RH=RH); s['T'] = T; s['P'] = P; return s

def state_from_DB_WB(Tdb_C, Twb_C, P=ATM_P):
    T = Tdb_C + 273.15
    Twb = min(Twb_C, Tdb_C) + 273.15
    # invert Twb(T,W) = target  ->  find W
    def fW(W): return HAPropsSI('Twb','T',T,'P',P,'W',W) - Twb
    W_lo = 1e-7
    try:
        W_hi = HAPropsSI('W','T',T,'P',P,'R',0.999)
    except Exception:
        W_hi = 0.03
    try:
        W = bisect_solve(fW, W_lo, W_hi, tol=1e-8)
    except Exception:
        W = HAPropsSI('W','T',T,'P',P,'R',0.5)
    s = humid_air_props(T, P, W=W); s['T'] = T; s['P'] = P; return s



def mix_air_streams(streams, P=ATM_P):
    """Mix humid-air streams adiabatically on a dry-air basis.

    Each stream is a dict with keys: state, Vdot_m3s, name.
    Returns mixed state, total dry/moist mass flows, and component flow details.
    """
    if not streams:
        raise ValueError("At least one air stream is required.")

    details = []
    mda_total = 0.0
    mma_total = 0.0
    water_total = 0.0
    H_total = 0.0

    for item in streams:
        s = item["state"]
        V = float(item["Vdot_m3s"])
        if V < 0:
            raise ValueError("Volumetric flow cannot be negative.")
        m_moist = s["rho"] * V
        m_da = m_moist / (1.0 + s["W"])
        m_w = m_da * s["W"]
        Hdot = m_da * s["h"]
        mda_total += m_da
        mma_total += m_moist
        water_total += m_w
        H_total += Hdot
        details.append({
            "name": item.get("name", "Stream"), "Vdot_m3s": V,
            "m_dot_moist": m_moist, "m_dot_dry": m_da,
            "m_dot_water": m_w, "Hdot_W": Hdot, "state": s,
        })

    if mda_total <= 1e-12:
        raise ValueError("Total dry-air mass flow must be greater than zero.")

    W_mix = water_total / mda_total
    h_mix = H_total / mda_total

    def fT(T):
        return HAPropsSI('H', 'T', T, 'P', P, 'W', W_mix) - h_mix

    Tmin = min(d["state"]["T"] for d in details) - 30.0
    Tmax = max(d["state"]["T"] for d in details) + 30.0
    T_mix = bisect_solve(fT, max(173.15, Tmin), min(373.15, Tmax), tol=1e-8)
    s_mix = humid_air_props(T_mix, P, W=W_mix)
    s_mix["T"] = T_mix
    s_mix["P"] = P
    V_mix = (1.0 + W_mix) * mda_total / max(s_mix["rho"], 1e-12)

    return {
        "state": s_mix,
        "m_dot_dry": mda_total,
        "m_dot_moist": mma_total,
        "m_dot_water": water_total,
        "Hdot_W": H_total,
        "Vdot_m3s": V_mix,
        "details": details,
    }

# ---------- Process solver (Q̇ on flowing air) ----------

# ---------- Coil capacity from inlet/outlet DB+WB ----------
def coil_capacity_from_in_out(
    Tdb_in_C, Twb_in_C,
    Tdb_out_C, Twb_out_C,
    Vdot_m3s,                 # volumetric flow (m3/s)
    flow_measured_at="outlet",# "inlet" or "outlet"
    P=ATM_P
):
    """
    Coil capacity from inlet/outlet DB+WB and airflow.

    - Uses CoolProp HAPropsSI via state_from_DB_WB() for robust psychrometrics.
    - Handles airflow measured at inlet or outlet by converting V̇ to ṁ using density at that state.
    - Returns sensible/latent/total kW, SHR, TR, and state details.
    """
    s_in  = state_from_DB_WB(Tdb_in_C,  Twb_in_C,  P)
    s_out = state_from_DB_WB(Tdb_out_C, Twb_out_C, P)

    # Convert V̇ (measured at inlet or outlet) -> moist-air mass flow using that state's density
    s_ref = s_out if str(flow_measured_at).lower().startswith("out") else s_in
    m_dot_moist = s_ref["rho"] * Vdot_m3s  # kg moist air / s

    # Convert to dry-air mass flow using the same reference state's humidity ratio
    m_dot_dry = m_dot_moist / (1.0 + s_ref["W"])  # kg dry air / s

    # Total: ṁ_da * Δh (h is J/kg_da in this app)
    Q_total_kW = m_dot_dry * (s_in["h"] - s_out["h"]) / 1000.0

    # Sensible: ṁ_da * cp_in * ΔTdb  (cp in J/kg_da-K)
    cp_in = s_in["cp"]
    Q_sens_kW  = m_dot_dry * cp_in * (Tdb_in_C - Tdb_out_C) / 1000.0

    Q_lat_kW = Q_total_kW - Q_sens_kW
    SHR = Q_sens_kW / Q_total_kW if abs(Q_total_kW) > 1e-12 else float("nan")

    return {
        "s_in": s_in,
        "s_out": s_out,
        "m_dot_dry": m_dot_dry,
        "Q_sens_kW": Q_sens_kW,
        "Q_lat_kW": Q_lat_kW,
        "Q_total_kW": Q_total_kW,
        "SHR": SHR,
        "TR_total": Q_total_kW / 3.517,
        "TR_sens":  Q_sens_kW / 3.517,
        "TR_lat":   Q_lat_kW / 3.517,
    }


def final_state_after_Qdot(s1, Qdot_kW, m_dot_dry, P=ATM_P):
    """Return (s2, note, condensate_dict|None). Uses dry-air mass flow for balances."""
    if abs(Qdot_kW) < 1e-12 or m_dot_dry <= 1e-12:
        return s1.copy(), "No change (zero heat or zero mass flow).", None

    h1, W1, T1 = s1['h'], s1['W'], s1['T']
    q_per_kg_dry = (Qdot_kW*1000.0) / m_dot_dry  # J per kg_dry
    h2_target = h1 + q_per_kg_dry

    # Dry (constant-W) attempt: solve H(T,W1) == h2_target
    def H_constW(T): return HAPropsSI('H','T',T,'P',P,'W',W1) - h2_target
    T_lo = max(173.15, T1 - 100.0); T_hi = min(373.15, T1 + 100.0)
    try:
        T2_dry = bisect_solve(H_constW, T_lo, T_hi)
        RH2 = HAPropsSI('R','T',T2_dry,'P',P,'W',W1)
        if RH2 <= 0.999 or q_per_kg_dry >= 0:
            s2 = humid_air_props(T2_dry, P, W=W1); s2['T'] = T2_dry
            return s2, "Dry process (W constant).", None
    except Exception:
        pass

    # Saturated (condensation) solution: H(T, Wsat(T)) == target
    def H_sat(T):
        Wsat = HAPropsSI('W','T',T,'P',P,'R',0.999)
        return HAPropsSI('H','T',T,'P',P,'W',Wsat) - h2_target
    try:
        T2 = bisect_solve(H_sat, max(173.15, T1-80.0), T1)
        W2 = HAPropsSI('W','T',T2,'P',P,'R',0.999)
        s2 = humid_air_props(T2, P, W=W2); s2['T'] = T2
        dW = max(0.0, W1 - W2)                 # kg_vapor/kg_dry removed
        mdot_cond_kg_s = dW * m_dot_dry        # kg/s
        condensate = {
            'dW_g_per_kg': 1000.0*dW,
            'mdot_g_s': 1000.0*mdot_cond_kg_s,
            'mdot_kg_h': 3600.0*mdot_cond_kg_s,
            'vol_mL_s': 1000.0*mdot_cond_kg_s,
            'vol_L_h': 3.6*mdot_cond_kg_s
        }
        return s2, "Cooling with condensation (final state saturated).", condensate
    except Exception as e:
        # Last resort: constant-W sensible estimate
        cp1 = s1['cp']
        T2_guess = np.clip(T1 + (h2_target - h1)/max(cp1,1e-9), T_lo, T_hi)
        s2 = humid_air_props(T2_guess, P, W=W1); s2['T'] = T2_guess
        return s2, f"Dry fallback — check inputs. ({e})", None

# -------------------- display helpers --------------------
def enthalpy_dual(s):
    """Return (h_dry_kJkgda, h_moist_kJkg)."""
    h_dry_kJkgda = s['h']/1000.0
    h_moist_kJkg = h_dry_kJkgda / (1.0 + s['W'])
    return h_dry_kJkgda, h_moist_kJkg

def state_table(title, s, Vdot_in=None, m_dry=None, show_flows=True, show_outlet_vol=False):
    T_C   = s['T'] - 273.15
    Twb_C = s['Twb'] - 273.15
    Tdp_C = s['Tdp'] - 273.15
    v     = 1.0/max(s['rho'],1e-12)
    W_gkg = s['W']*1000.0
    h_dry, h_moist = enthalpy_dual(s)

    st.subheader(title)
    st.markdown("### Psychrometric state")
    st.write(f"**Dry-bulb (DB):** {T_C:.2f} °C")
    st.write(f"**Wet-bulb (WB):** {Twb_C:.2f} °C")
    st.write(f"**Dew-point (DP):** {Tdp_C:.2f} °C")
    st.write(f"**Relative humidity (RH):** {s['RH']*100:.2f} %")
    st.write(f"**Humidity ratio (W):** {W_gkg:.3f} g/kg dry air")
    st.write(f"**Enthalpy per kg dry air (hᵈʳʸ):** {h_dry:.3f} kJ/kg₍da₎")
    st.write(f"**Enthalpy per kg moist air (hᵐᵒᶦˢᵗ):** {h_moist:.3f} kJ/kg₍moist₎")

    st.markdown("### Transport & derived")
    st.write(f"**Density (ρ):** {s['rho']:.4f} kg/m³")
    st.write(f"**Specific volume (v):** {v:.4f} m³/kg₍da₎")
    st.write(f"**Specific heat (cp):** {s['cp']:.1f} J/kg·K")
    st.write(f"**Thermal conductivity (k):** {s['k']:.5f} W/m·K")
    st.write(f"**Dynamic viscosity (μ):** {s['mu']:.7f} Pa·s")
    st.write(f"**Prandtl number (Pr):** {s['Pr']:.3f}")
    st.write(f"**Kinematic viscosity (ν):** {s['nu']:.7e} m²/s")
    st.write(f"**Thermal diffusivity (α):** {s['alpha']:.7e} m²/s")

    if show_flows and Vdot_in is not None and m_dry is not None:
        m_moist = s['rho'] * Vdot_in
        water_g_s = s['W'] * m_dry * 1000.0
        st.markdown("### Flow conversions (based on this state's density)")
        st.write(f"**Volumetric flow (V̇_air):** {Vdot_in:.3f} m³/s")
        st.write(f"**Moist-air mass flow (ṁ_moist ≈ ρ·V̇):** {m_moist:.3f} kg/s")
        st.write(f"**Dry-air mass flow (ṁ_dry = ṁ_moist/(1+W)):** {m_dry:.3f} kg₍da₎/s")
        st.write(f"**Water mass flow in stream (ṁ_w = W·ṁ_dry):** {water_g_s:.2f} g/s")

    if show_outlet_vol and m_dry is not None:
        Vdot_state = (1.0 + s['W']) * m_dry / max(s['rho'], 1e-12)
        st.write(f"**Implied volumetric flow at this state:** {Vdot_state:.3f} m³/s")

def process_capacity_block(s1, s2, m_dot_dry):
    h1 = s1['h']; h2 = s2['h']  # J/kg_da
    q_kW = m_dot_dry * (h2 - h1) / 1000.0  # kW
    st.markdown("### Process capacity from enthalpy change")
    st.write(f"**Total capacity (Q̇ = ṁ₍da₎·Δh):** {q_kW:.3f} kW")
    hdry1, _ = enthalpy_dual(s1)
    hdry2, _ = enthalpy_dual(s2)
    Hdot1 = m_dot_dry * hdry1  # kW (kJ/s)
    Hdot2 = m_dot_dry * hdry2
    st.write(f"**Enthalpy flow at inlet:** {Hdot1:.3f} kW  (per kg₍da₎ basis)")
    st.write(f"**Enthalpy flow at outlet:** {Hdot2:.3f} kW  (per kg₍da₎ basis)")
    return q_kW

# -------------------- PDF builder (Unicode-aware) --------------------
def _latin1_sanitize(s: str) -> str:
    repl = {
        "ρ":"rho", "μ":"mu", "ν":"nu", "α":"alpha",
        "₍":"(", "₎":")", "₋":"-", "–":"-", "—":"-",
        "≈":"~", "’":"'","“":'"',"”":'"',
        "₍da₎":"(da)", "₍moist₎":"(moist)",
    }
    for k,v in repl.items():
        s = s.replace(k, v)
    try:
        s.encode("latin-1")
        return s
    except UnicodeEncodeError:
        return s.encode("latin-1", "ignore").decode("latin-1")

def _kv_to_printable(d: dict, latin1_mode: bool) -> dict:
    out = {}
    for k, v in d.items():
        ks = str(k); vs = str(v)
        if latin1_mode:
            ks = _latin1_sanitize(ks)
            vs = _latin1_sanitize(v)
        out[ks] = vs
    return out

def build_pdf(data, font_path: str | None = None):
    """
    data keys: title, logo_bytes|None, notes_text, sections (list),
               inputs, inlet, outlet, flows, capacity, condensate
    font_path: optional path to a Unicode TTF (e.g., DejaVuSans.ttf)
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    latin1_mode = font_path is None
    if not latin1_mode:
        try:
            pdf.add_font("UNI", "", font_path, uni=True)
            pdf.set_font("UNI", "", 16)
        except Exception:
            latin1_mode = True
    if latin1_mode:
        pdf.set_font("Arial", "B", 16)

    title = data.get("title", "Moist Air Report")
    if latin1_mode: title = _latin1_sanitize(title)
    pdf.cell(0, 10, title, 0, 1, "C")

    if data.get("logo_bytes"):
        try:
            tmp_logo = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            raw = data["logo_bytes"].getvalue() if isinstance(data["logo_bytes"], BytesIO) else data["logo_bytes"]
            tmp_logo.write(raw); tmp_logo.flush()
            pdf.image(tmp_logo.name, x=170, y=10, w=25)
        except Exception:
            pass

    def section_header(txt):
        if latin1_mode:
            txt = _latin1_sanitize(txt); pdf.set_font("Arial", "B", 12)
        else:
            pdf.set_font("UNI", "", 12)
        pdf.ln(2); pdf.cell(0, 8, txt, 0, 1)

    def kv_table(d):
        d = _kv_to_printable(d, latin1_mode)
        if latin1_mode: pdf.set_font("Arial", "", 10)
        else:           pdf.set_font("UNI", "", 10)
        for k, v in d.items():
            pdf.cell(85, 6, f"{k}:", 0, 0)
            pdf.cell(0, 6, f"{v}", 0, 1)

    for sec in data.get("sections", []):
        if sec == "Inputs":
            section_header("Inputs"); kv_table(data.get("inputs", {}))
        elif sec == "Return air":
            section_header("Return air"); kv_table(data.get("return_air", {}))
        elif sec == "Fresh air":
            section_header("Fresh air"); kv_table(data.get("fresh_air", {}))
        elif sec == "Mixed air":
            section_header("Mixed air entering coil"); kv_table(data.get("mixed_air", {}))
        elif sec == "Inlet state":
            section_header("Inlet state"); kv_table(data.get("inlet", {}))
        elif sec == "Outlet state":
            section_header("Outlet state"); kv_table(data.get("outlet", {}))
        elif sec == "Flows & rates":
            section_header("Flows & rates"); kv_table(data.get("flows", {})); kv_table(data.get("capacity", {}))
        elif sec == "Condensate":
            if data.get("condensate"):
                section_header("Condensate"); kv_table(data.get("condensate", {}))
        elif sec == "Notes":
            notes = data.get("notes_text","").strip()
            if notes:
                section_header("Notes")
                txt = _latin1_sanitize(notes) if latin1_mode else notes
                if latin1_mode: pdf.set_font("Arial", "", 10)
                else:           pdf.set_font("UNI", "", 10)
                for line in txt.splitlines():
                    pdf.multi_cell(0, 6, line)

    out = pdf.output(dest="S")
    return out if isinstance(out, (bytes, bytearray)) else out.encode("latin-1")

# -------------------- UI --------------------
st.title("Moist Air Calculator — Mixed Air + Heating/Cooling Process")
st.caption("Mix return air and fresh air, then apply a direct heat rate to the mixed stream. Use negative kW for cooling and positive kW for heating.")

calc_mode = st.sidebar.radio(
    "Calculator",
    ["Mixed-air process Q̇", "Single-stream process Q̇", "Coil capacity from In/Out DB+WB"],
    index=0,
    key="calc_mode_radio"
)

# Moisture-mode selectors stay outside the form so the form updates immediately.
if calc_mode == "Mixed-air process Q̇":
    st.sidebar.markdown("### Return air")
    ra_mode = st.sidebar.radio("Return-air moisture input", ["DB + RH", "DB + WB"], key="ra_mode")
    st.sidebar.markdown("### Fresh air")
    fa_mode = st.sidebar.radio("Fresh-air moisture input", ["DB + RH", "DB + WB"], key="fa_mode")
elif calc_mode == "Single-stream process Q̇":
    single_mode = st.sidebar.radio("Moisture input mode", ["DB + RH", "DB + WB"], key="single_mode")

with st.sidebar.form("inputs_form", clear_on_submit=False):
    st.header("General")
    P_mode = st.selectbox("Pressure mode", ["Sea level (101325 Pa)", "Custom (Pa)"], index=0)
    P_txt = text_num("Pressure (Pa)", "txt_P", "101325")

    if calc_mode == "Mixed-air process Q̇":
        st.header("Return air stream")
        ra_Tdb_txt = text_num("Return air DB (°C)", "ra_Tdb", "26.0")
        if st.session_state.get("ra_mode", "DB + RH") == "DB + RH":
            ra_RH_txt = text_num("Return air RH (%)", "ra_RH", "50.0"); ra_Twb_txt = None
        else:
            ra_Twb_txt = text_num("Return air WB (°C)", "ra_Twb", "19.0"); ra_RH_txt = None
        ra_flow_txt = text_num("Return airflow", "ra_flow", "6000")
        ra_units = st.selectbox("Return airflow units", ["m³/h", "m³/s"], key="ra_units")

        st.header("Fresh air stream")
        fa_Tdb_txt = text_num("Fresh air DB (°C)", "fa_Tdb", "45.0")
        if st.session_state.get("fa_mode", "DB + RH") == "DB + RH":
            fa_RH_txt = text_num("Fresh air RH (%)", "fa_RH", "35.0"); fa_Twb_txt = None
        else:
            fa_Twb_txt = text_num("Fresh air WB (°C)", "fa_Twb", "27.0"); fa_RH_txt = None
        fa_flow_txt = text_num("Fresh airflow", "fa_flow", "2500")
        fa_units = st.selectbox("Fresh airflow units", ["m³/h", "m³/s"], key="fa_units")

        st.header("Coil process")
        qdot_txt = text_num("External heat rate Q̇ (kW)", "txt_qdot", "-49.6", help="Negative = cooling; positive = heating")

    elif calc_mode == "Single-stream process Q̇":
        st.header("Air stream")
        Tdb_txt = text_num("Dry-bulb (°C)", "txt_Tdb", "30.0")
        if st.session_state.get("single_mode", "DB + RH") == "DB + RH":
            RH_txt = text_num("Relative Humidity (%)", "txt_RH", "50.0"); Twb_txt = None
        else:
            Twb_txt = text_num("Wet-bulb (°C)", "txt_Twb", "20.0"); RH_txt = None
        Vdot_txt = text_num("Volumetric airflow", "txt_vdot", "1.20")
        single_units = st.selectbox("Airflow units", ["m³/s", "m³/h"], key="single_units")
        qdot_txt = text_num("External heat rate Q̇ (kW)", "txt_qdot", "-5.0")

    st.header("PDF options")
    report_title = st.text_input("Report title", value="Moist Air Report", key="report_title")
    notes_text = st.text_area("Notes (optional)", height=100, key="notes_text")
    default_sections = ["Inputs", "Return air", "Fresh air", "Mixed air", "Outlet state", "Flows & rates", "Condensate", "Notes"] if calc_mode == "Mixed-air process Q̇" else ["Inputs", "Inlet state", "Outlet state", "Flows & rates", "Condensate", "Notes"]
    sections = st.multiselect(
        "PDF sections",
        ["Inputs", "Return air", "Fresh air", "Mixed air", "Inlet state", "Outlet state", "Flows & rates", "Condensate", "Notes"],
        default=default_sections,
        key="sections"
    )
    submitted = st.form_submit_button("Update / Calculate", use_container_width=True)

errors = []
try:
    P_val = parse_number(P_txt, min_val=50000, max_val=120000, field_name="Pressure")
    P = P_val if P_mode.startswith("Custom") else ATM_P
except ValueError as e:
    errors.append(str(e)); P = ATM_P

# Existing independent coil-capacity calculator
if calc_mode == "Coil capacity from In/Out DB+WB":
    st.header("Coil capacity from inlet/outlet DB + WB")
    c1, c2 = st.columns(2)
    with c1:
        Tdb_in2 = st.number_input("Inlet DB (°C)", value=24.3, step=0.1)
        Twb_in2 = st.number_input("Inlet WB (°C)", value=18.1, step=0.1)
    with c2:
        Tdb_out2 = st.number_input("Outlet DB (°C)", value=13.5, step=0.1)
        Twb_out2 = st.number_input("Outlet WB (°C)", value=12.8, step=0.1)
    flow_units = st.radio("Airflow units", ["m³/s", "m³/h"], horizontal=True, index=1)
    Vdot_in = st.number_input("Airflow", value=6250.0 if flow_units == "m³/h" else 6250.0/3600.0)
    Vdot2_m3s = Vdot_in/3600.0 if flow_units == "m³/h" else Vdot_in
    flow_meas2 = st.selectbox("Airflow measured at", ["outlet", "inlet"], index=0)
    try:
        cap = coil_capacity_from_in_out(Tdb_in2, min(Twb_in2,Tdb_in2), Tdb_out2, min(Twb_out2,Tdb_out2), Vdot2_m3s, flow_measured_at=flow_meas2, P=P)
        a,b,c = st.columns(3)
        a.metric("Sensible", f"{cap['Q_sens_kW']:.2f} kW")
        b.metric("Latent", f"{cap['Q_lat_kW']:.2f} kW")
        c.metric("Total", f"{cap['Q_total_kW']:.2f} kW")
        a,b,c = st.columns(3)
        a.metric("SHR", f"{cap['SHR']:.3f}")
        b.metric("Dry-air flow", f"{cap['m_dot_dry']:.3f} kg/s")
        c.metric("Total", f"{cap['TR_total']:.2f} TR")
    except Exception as e:
        st.error(f"Capacity calculation failed: {e}")
    st.stop()


def parse_state(prefix, mode_name, Tdb_text, RH_text, Twb_text):
    Tdb = parse_number(Tdb_text, min_val=-60, max_val=120, field_name=f"{prefix} DB")
    if mode_name == "DB + RH":
        RH = parse_number(RH_text, min_val=0.1, max_val=99.9, field_name=f"{prefix} RH")
        return state_from_DB_RH(Tdb, RH, P), Tdb, RH, None
    Twb = parse_number(Twb_text, min_val=-60, max_val=120, field_name=f"{prefix} WB")
    if Twb > Tdb:
        raise ValueError(f"{prefix} WB cannot exceed DB.")
    return state_from_DB_WB(Tdb, Twb, P), Tdb, None, Twb

try:
    Qdot_kW = parse_number(qdot_txt, min_val=-10000, max_val=10000, field_name="Heat rate")
except ValueError as e:
    errors.append(str(e)); Qdot_kW = 0.0

return_air = fresh_air = mixed = None
if calc_mode == "Mixed-air process Q̇":
    try:
        s_ra, ra_Tdb, ra_RH, ra_Twb = parse_state("Return air", st.session_state.get("ra_mode","DB + RH"), ra_Tdb_txt, ra_RH_txt, ra_Twb_txt)
        s_fa, fa_Tdb, fa_RH, fa_Twb = parse_state("Fresh air", st.session_state.get("fa_mode","DB + RH"), fa_Tdb_txt, fa_RH_txt, fa_Twb_txt)
        ra_flow = parse_number(ra_flow_txt, min_val=0, max_val=2_000_000, field_name="Return airflow")
        fa_flow = parse_number(fa_flow_txt, min_val=0, max_val=2_000_000, field_name="Fresh airflow")
        ra_V = ra_flow/3600.0 if ra_units == "m³/h" else ra_flow
        fa_V = fa_flow/3600.0 if fa_units == "m³/h" else fa_flow
        mixed = mix_air_streams([
            {"name":"Return air", "state":s_ra, "Vdot_m3s":ra_V},
            {"name":"Fresh air", "state":s_fa, "Vdot_m3s":fa_V},
        ], P)
        s1 = mixed["state"]
        m_dot_dry = mixed["m_dot_dry"]
        Vdot_m3s = mixed["Vdot_m3s"]
        m_dot_moist_in = (1+s1["W"])*m_dot_dry
        return_air, fresh_air = s_ra, s_fa
    except ValueError as e:
        errors.append(str(e))
else:
    try:
        s1, Tdb_C, RH_pct, Twb_C = parse_state("Air", st.session_state.get("single_mode","DB + RH"), Tdb_txt, RH_txt, Twb_txt)
        Vraw = parse_number(Vdot_txt, min_val=0, max_val=2_000_000, field_name="Airflow")
        Vdot_m3s = Vraw/3600.0 if single_units == "m³/h" else Vraw
        m_dot_moist_in = s1["rho"]*Vdot_m3s
        m_dot_dry = m_dot_moist_in/(1+s1["W"])
    except ValueError as e:
        errors.append(str(e))

if errors:
    st.error("Please correct the following inputs:")
    for e in errors: st.write("• " + e)
    st.stop()

s2, note, condensate = final_state_after_Qdot(s1, Qdot_kW, m_dot_dry, P)
Vdot_out = (1+s2["W"])*m_dot_dry/max(s2["rho"],1e-12)

if calc_mode == "Mixed-air process Q̇":
    st.markdown("## Air mixing")
    c1,c2 = st.columns(2)
    with c1: state_table("Return Air", return_air, Vdot_in=ra_V, m_dry=mixed['details'][0]['m_dot_dry'])
    with c2: state_table("Fresh Air", fresh_air, Vdot_in=fa_V, m_dry=mixed['details'][1]['m_dot_dry'])
    st.markdown("## Coil process")
    c1,c2 = st.columns(2)
    with c1: state_table("Mixed Air Entering Coil", s1, Vdot_in=Vdot_m3s, m_dry=m_dot_dry)
    with c2:
        state_table("Final Air Leaving Coil", s2, m_dry=m_dot_dry, show_flows=False, show_outlet_vol=True)
        st.info(note)
else:
    c1,c2=st.columns(2)
    with c1: state_table("Initial State", s1, Vdot_in=Vdot_m3s, m_dry=m_dot_dry)
    with c2:
        state_table("Final State", s2, m_dry=m_dot_dry, show_flows=False, show_outlet_vol=True)
        st.info(note)

st.markdown("## Process results")
q_kW = process_capacity_block(s1, s2, m_dot_dry)
Q_total_cooling = max(0.0, -q_kW)
if Q_total_cooling > 0:
    Q_sens = m_dot_dry*s1['cp']*((s1['T']-s2['T']))/1000.0
    Q_sens = max(0.0, min(Q_total_cooling, Q_sens))
    Q_lat = Q_total_cooling-Q_sens
    a,b,c,d=st.columns(4)
    a.metric("Total cooling", f"{Q_total_cooling:.2f} kW")
    b.metric("Sensible cooling", f"{Q_sens:.2f} kW")
    c.metric("Latent cooling", f"{Q_lat:.2f} kW")
    d.metric("SHR", f"{Q_sens/Q_total_cooling:.3f}")

if condensate:
    st.markdown("## Condensate")
    a,b,c=st.columns(3)
    a.metric("Moisture removed", f"{condensate['dW_g_per_kg']:.3f} g/kgda")
    b.metric("Condensate", f"{condensate['mdot_kg_h']:.3f} kg/h")
    c.metric("Condensate volume", f"{condensate['vol_L_h']:.3f} L/h")

# PDF preparation
def state_dict(s):
    return {
        "DB / WB / DP (°C)": f"{s['T']-273.15:.2f} / {s['Twb']-273.15:.2f} / {s['Tdp']-273.15:.2f}",
        "RH (%)": f"{100*s['RH']:.2f}",
        "W (g/kg_da)": f"{1000*s['W']:.3f}",
        "Enthalpy dry | moist": f"{s['h']/1000:.3f} | {s['h']/1000/(1+s['W']):.3f} kJ/kg",
        "Density (kg/m3)": f"{s['rho']:.4f}",
    }

if calc_mode == "Mixed-air process Q̇":
    inputs_dict = {
        "Pressure": f"{P:.0f} Pa", "Process heat rate": f"{Qdot_kW:.3f} kW",
        "Return airflow": f"{ra_V:.4f} m3/s ({ra_V*3600:.1f} m3/h)",
        "Fresh airflow": f"{fa_V:.4f} m3/s ({fa_V*3600:.1f} m3/h)",
        "Mixed airflow": f"{Vdot_m3s:.4f} m3/s ({Vdot_m3s*3600:.1f} m3/h)",
    }
else:
    inputs_dict = {"Pressure":f"{P:.0f} Pa", "Airflow":f"{Vdot_m3s:.4f} m3/s", "Process heat rate":f"{Qdot_kW:.3f} kW"}

flows_dict = {
    "Dry-air mass flow": f"{m_dot_dry:.4f} kg_da/s",
    "Moist-air mass flow at coil inlet": f"{m_dot_moist_in:.4f} kg/s",
    "Coil inlet volume": f"{Vdot_m3s:.4f} m3/s",
    "Coil outlet volume": f"{Vdot_out:.4f} m3/s",
}
capacity_dict = {"Q from enthalpy":f"{q_kW:.3f} kW", "Process note":note}
if Q_total_cooling>0:
    capacity_dict.update({"Sensible cooling":f"{Q_sens:.3f} kW", "Latent cooling":f"{Q_lat:.3f} kW", "SHR":f"{Q_sens/Q_total_cooling:.3f}"})
cond_dict = None if not condensate else {
    "Delta W":f"{condensate['dW_g_per_kg']:.3f} g/kg_da", "Condensate":f"{condensate['mdot_kg_h']:.3f} kg/h", "Volume":f"{condensate['vol_L_h']:.3f} L/h"
}

st.markdown("## PDF report")
c1,c2=st.columns(2)
with c1: logo_u2=st.file_uploader("Logo", type=["png","jpg","jpeg"], key="logo_u2")
with c2: font_u2=st.file_uploader("Unicode font (optional)", type=["ttf","otf"], key="font_u2")
logo_bytes=BytesIO(logo_u2.read()) if logo_u2 else None
font_path=None
if font_u2:
    tf=tempfile.NamedTemporaryFile(delete=False,suffix=".ttf"); tf.write(font_u2.read()); tf.flush(); font_path=tf.name
pdf_bytes=build_pdf({
    "title":report_title, "logo_bytes":logo_bytes, "notes_text":notes_text, "sections":sections,
    "inputs":inputs_dict, "return_air":state_dict(return_air) if return_air else {},
    "fresh_air":state_dict(fresh_air) if fresh_air else {}, "mixed_air":state_dict(s1) if mixed else {},
    "inlet":state_dict(s1), "outlet":state_dict(s2), "flows":flows_dict,
    "capacity":capacity_dict, "condensate":cond_dict,
}, font_path=font_path)
if isinstance(pdf_bytes,str): pdf_bytes=pdf_bytes.encode('latin-1','ignore')
st.download_button("📄 Download PDF report", data=bytes(pdf_bytes), file_name="moist_air_mixed_stream_report.pdf", mime="application/pdf", use_container_width=True)

with st.expander("Mixing balance check"):
    if mixed:
        st.write({
            "Return dry-air flow kg/s": mixed['details'][0]['m_dot_dry'],
            "Fresh dry-air flow kg/s": mixed['details'][1]['m_dot_dry'],
            "Total dry-air flow kg/s": mixed['m_dot_dry'],
            "Mixed humidity ratio kg/kg_da": s1['W'],
            "Mixed enthalpy kJ/kg_da": s1['h']/1000,
        })

st.caption("Mixed-air calculations use dry-air mass, water-vapour mass and enthalpy conservation. Each entered volumetric flow is converted using that stream's own density.")
