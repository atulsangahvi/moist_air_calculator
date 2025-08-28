# Moist Air Calculator — Streamlit (named outputs + water content + condensate)
# Inputs: DB + (RH or WB), pressure; m_dot_air, Q̇
# Outputs:
#  • Clear named properties (with symbols & units)
#  • Water content per kg dry air (g/kg_da) and water mass flow (g/s)
#  • If condensation occurs: condensate removed (g/kg_da, g/s, kg/h, mL/s, L/h)

import streamlit as st
import numpy as np
from scipy.optimize import root_scalar
from CoolProp.CoolProp import HAPropsSI

ATM_P = 101325.0

# ---------- Robust humid-air helpers ----------
def humid_air_props(T_K, P_Pa=ATM_P, RH=None, W=None):
    """
    Return humid-air properties at (T, P) using either RH or humidity ratio W.
    Keys:
      T[K], RH[-], W[kg/kg_da], h[J/kg_da], Tdp[K], Twb[K],
      rho[kg/m3], mu[Pa·s], k[W/m·K], cp[J/kg·K], Pr[-], nu[m2/s], alpha[m2/s]
    """
    if RH is None and W is None:
        raise ValueError("Provide either RH or W")

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

    # Density
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

    # Secondary
    Pr = cp*mu/max(k, 1e-12)
    nu = mu/max(rho, 1e-12)
