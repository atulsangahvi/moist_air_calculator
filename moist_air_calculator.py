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

    # sa
