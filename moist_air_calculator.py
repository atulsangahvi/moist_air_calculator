# Moist Air Calculator — HVAC Volumetric Flow (m³/s) + Dual Enthalpy + Unicode-Safe PDF
# - No SciPy required (pure-Python bisection for root-finding)
# - Pin Python 3.11 (via runtime.txt) so CoolProp/Numpy wheels are available

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
    s = humid_air_props(T, P, RH=RH); s['T'] = T; return s

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
    s = humid_air_props(T, P, W=W); s['T'] = T; return s

# ---------- Process solver (Q̇ on flowing air) ----------
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
            vs = _latin1_sanitize(vs)
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
st.title("Moist Air Calculator — HVAC Volumetric Flow (m³/s) + PDF")

with st.sidebar.form("inputs_form", clear_on_submit=False):
    st.header("Inputs")

    P_mode = st.selectbox("Pressure mode", ["Sea level (101325 Pa)", "Custom (Pa)"], index=0)
    P_txt  = text_num("Pressure (Pa)", "txt_P", "101325", help="Used if 'Custom' is selected.")

    mode = st.radio("Moisture input mode", ["DB + RH", "DB + WB"], index=0)
    Tdb_txt = text_num("Dry-bulb (°C)", "txt_Tdb", "30.0")

    if mode == "DB + RH":
        RH_txt  = text_num("Relative Humidity (%)", "txt_RH", "50.0")
        Twb_txt = None
    else:
        Twb_txt = text_num("Wet-bulb (°C)", "txt_Twb", "20.0")
        RH_txt  = None

    st.header("Process (Q̇ on flowing air)")
    Vdot_txt = text_num("Volumetric flow V̇_air (m³/s) — at inlet", "txt_vdot", "1.20")
    qdot_txt = text_num("External heat rate Q̇ (kW)  (+heating / −cooling)", "txt_qdot", "-5.0")

    st.header("PDF options")
    report_title = st.text_input("Report title", value="Moist Air Report")
    logo_file = st.file_uploader("Logo (PNG/JPG)", type=["png","jpg","jpeg"])
    notes_text = st.text_area("Notes (optional)", height=120, placeholder="Type comments/headings here...")
    sections = st.multiselect(
        "Select sections (click in your desired order)",
        ["Inputs","Inlet state","Outlet state","Flows & rates","Condensate","Notes"],
        default=["Inputs","Inlet state","Outlet state","Flows & rates","Condensate","Notes"],
    )
    font_file = st.file_uploader("Custom PDF font (TTF/OTF) — e.g., DejaVuSans.ttf / NotoSans-Regular.ttf", type=["ttf","otf"])

    submitted = st.form_submit_button("Update / Calculate", use_container_width=True)

# ---- parse & validate ----
errors = []

try:
    P_val = parse_number(P_txt, min_val=50000, max_val=120000, field_name="Pressure")
    P = P_val if P_mode.startswith("Custom") else ATM_P
except ValueError as e:
    errors.append(str(e)); P = ATM_P

try:
    Tdb_C = parse_number(Tdb_txt, min_val=-60.0, max_val=120.0, field_name="Dry-bulb")
except ValueError as e:
    errors.append(str(e)); Tdb_C = 30.0

RH_pct = None; Twb_C = None
if mode == "DB + RH":
    try:
        RH_pct = parse_number(RH_txt, min_val=1.0, max_val=99.0, field_name="Relative Humidity (%)")
    except ValueError as e:
        errors.append(str(e)); RH_pct = 50.0
else:
    try:
        Twb_C = parse_number(Twb_txt, min_val=-60.0, max_val=120.0, field_name="Wet-bulb")
    except ValueError as e:
        errors.append(str(e)); Twb_C = 20.0
    if Twb_C > Tdb_C:
        errors.append("Wet-bulb cannot exceed Dry-bulb. It will be clamped to DB.")
        Twb_C = Tdb_C

try:
    Vdot_m3s = parse_number(Vdot_txt, min_val=0.0, max_val=500.0, field_name="Volumetric flow V̇_air")
except ValueError as e:
    errors.append(str(e)); Vdot_m3s = 1.2

try:
    Qdot_kW = parse_number(qdot_txt, min_val=-10000.0, max_val=10000.0, field_name="Heat rate")
except ValueError as e:
    errors.append(str(e)); Qdot_kW = -5.0

if errors:
    st.error("Please fix these inputs:")
    for e in errors: st.write("• " + e)

# ---- compute inlet state ----
s1 = state_from_DB_RH(Tdb_C, RH_pct, P) if RH_pct is not None else state_from_DB_WB(Tdb_C, Twb_C, P)

# Convert HVAC volumetric flow -> mass flows (inlet basis)
rho1 = s1['rho']; W1 = s1['W']
m_dot_moist_in = rho1 * Vdot_m3s                # kg moist air / s
m_dot_dry       = m_dot_moist_in / (1.0 + W1)   # kg dry air / s (used for balances)

# ---- solve outlet state ----
s2, note, condensate = final_state_after_Qdot(s1, Qdot_kW, m_dot_dry, P)

# implied outlet volumetric flow (density & W at outlet)
Vdot_out = (1.0 + s2['W']) * m_dot_dry / max(s2['rho'], 1e-12)

# ---- display ----
col1, col2 = st.columns(2)
with col1:
    state_table("Initial State", s1, Vdot_in=Vdot_m3s, m_dry=m_dot_dry, show_flows=True, show_outlet_vol=False)
with col2:
    state_table("Final State (after Q̇)", s2, Vdot_in=Vdot_m3s, m_dry=m_dot_dry, show_flows=True, show_outlet_vol=True)
    st.write(f"**Implied outlet volumetric flow (V̇_out):** {Vdot_out:.3f} m³/s")
    st.info(note)

st.markdown("## Capacity")
q_kW = process_capacity_block(s1, s2, m_dot_dry)

if condensate is not None:
    st.markdown("## Condensate (due to cooling)")
    st.write(f"**Water removed per kg dry air (ΔW):** {condensate['dW_g_per_kg']:.2f} g/kg₍da₎")
    st.write(f"**Condensate mass flow:** {condensate['mdot_g_s']:.2f} g/s ({condensate['mdot_kg_h']:.3f} kg/h)")
    st.write(f"**Condensate volume flow (≈ water):** {condensate['vol_mL_s']:.1f} mL/s ({condensate['vol_L_h']:.3f} L/h)")

with st.expander("App diagnostics", expanded=False):
    st.write({"CoolProp": getattr(sys.modules.get('CoolProp'), '__version__', 'unknown'),
              "Python": sys.version.split()[0],
              "Platform": platform.platform()})

# ---- Build PDF dicts ----
def enthalpy_pair(s):
    a,b = enthalpy_dual(s); return f"{a:.3f} kJ/kg₍da₎ | {b:.3f} kJ/kg₍moist₎"

inputs_dict = {
    "Pressure": f"{P:.0f} Pa",
    "Mode": "DB + RH" if RH_pct is not None else "DB + WB",
    "Dry-bulb (°C)": f"{Tdb_C:.2f}",
    "Relative Humidity (%)": f"{RH_pct:.2f}" if RH_pct is not None else "—",
    "Wet-bulb (°C)": f"{Twb_C:.2f}" if Twb_C is not None else "—",
    "Volumetric flow V̇_in": f"{Vdot_m3s:.3f} m³/s",
    "External heat rate Q̇": f"{Qdot_kW:.3f} kW",
}
inlet_dict = {
    "DB / WB / DP (°C)": f"{s1['T']-273.15:.2f} / {s1['Twb']-273.15:.2f} / {s1['Tdp']-273.15:.2f}",
    "RH (%)": f"{s1['RH']*100:.2f}",
    "W (g/kg₍da₎)": f"{s1['W']*1000.0:.3f}",
    "Enthalpy (dry | moist)": enthalpy_pair(s1),
    "ρ (kg/m³)": f"{s1['rho']:.4f}",
}
outlet_dict = {
    "DB / WB / DP (°C)": f"{s2['T']-273.15:.2f} / {s2['Twb']-273.15:.2f} / {s2['Tdp']-273.15:.2f}",
    "RH (%)": f"{s2['RH']*100:.2f}",
    "W (g/kg₍da₎)": f"{s2['W']*1000.0:.3f}",
    "Enthalpy (dry | moist)": enthalpy_pair(s2),
    "ρ (kg/m³)": f"{s2['rho']:.4f}",
}
flows_dict = {
    "ṁ_moist (kg/s) @ inlet": f"{m_dot_moist_in:.3f}",
    "ṁ_dry (kg₍da₎/s)": f"{m_dot_dry:.3f}",
    "V̇_in (m³/s)": f"{Vdot_m3s:.3f}",
    "V̇_out implied (m³/s)": f"{Vdot_out:.3f}",
}
hdry1,_ = enthalpy_dual(s1); hdry2,_ = enthalpy_dual(s2)
capacity_dict = {
    "Q̇ from Δh (kW)": f"{q_kW:.3f}",
    "Ḣ_in (kW)": f"{(m_dot_dry*hdry1):.3f}",
    "Ḣ_out (kW)": f"{(m_dot_dry*hdry2):.3f}",
}
cond_dict = None
if condensate is not None:
    cond_dict = {
        "ΔW (g/kg₍da₎)": f"{condensate['dW_g_per_kg']:.2f}",
        "ṁ_cond (g/s)": f"{condensate['mdot_g_s']:.2f}",
        "ṁ_cond (kg/h)": f"{condensate['mdot_kg_h']:.3f}",
        "V̇_cond (mL/s)": f"{condensate['vol_mL_s']:.1f}",
        "V̇_cond (L/h)": f"{condensate['vol_L_h']:.3f}",
    }

# Optional font
font_path = None
font_file = st.session_state.get("font_file_widget", None)  # just to avoid linter noise
# The actual uploader lives in the sidebar form above; re-read here:
# (Streamlit re-runs; we reopen the uploaded file only when building PDF)
with st.sidebar:
    pass

# Prepare optional font path (saved to a temp file if uploaded)
# We need to re-access the uploaded file from the form widget:
for k in st.session_state:
    pass
# Actually, we already consumed logo/font inside the form; re-open now:
# Streamlit keeps `font_file` object alive; we can use it directly:
# Build logo bytes
logo_bytes = None
# We can't reuse logo_file beyond the form scope reliably; offer a second uploader outside if needed.
# To keep it simple, ask user to upload again if logo missing on PDF.

# Safer approach: keep the controls inside the form and build PDF immediately after
# re-asking for font and logo objects:
st.markdown("## PDF")
colA, colB = st.columns(2)
with colA:
    logo_u2 = st.file_uploader("Logo (PNG/JPG) for PDF export", type=["png","jpg","jpeg"], key="logo_u2")
with colB:
    font_u2 = st.file_uploader("Unicode font (TTF/OTF) for PDF export", type=["ttf","otf"], key="font_u2")

if logo_u2 is not None:
    try:
        logo_bytes = BytesIO(logo_u2.read())
    except Exception:
        logo_bytes = None

if font_u2 is not None:
    try:
        tmp_font = tempfile.NamedTemporaryFile(delete=False, suffix=".ttf")
        tmp_font.write(font_u2.read()); tmp_font.flush()
        font_path = tmp_font.name
    except Exception:
        font_path = None

pdf_bytes = build_pdf({
    "title": st.session_state.get("report_title", "Moist Air Report") if "report_title" in st.session_state else "Moist Air Report",
    "logo_bytes": logo_bytes,
    "notes_text": st.session_state.get("notes_text", "") if "notes_text" in st.session_state else "",
    "sections": st.session_state.get("sections", ["Inputs","Inlet state","Outlet state","Flows & rates","Condensate","Notes"]),
    "inputs": inputs_dict,
    "inlet": inlet_dict,
    "outlet": outlet_dict,
    "flows": flows_dict,
    "capacity": capacity_dict,
    "condensate": cond_dict,
}, font_path=font_path)

st.download_button("📄 Download PDF report", data=pdf_bytes,
                   file_name="moist_air_report.pdf", mime="application/pdf",
                   use_container_width=True)

st.caption(
    "Pinned to Python 3.11 to ensure binary wheels install. "
    "No SciPy required. Upload a Unicode TTF for full symbol support in the PDF; "
    "otherwise the app auto-sanitizes to ASCII-safe text."
)
