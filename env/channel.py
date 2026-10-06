"""Path loss and rate. Internal units: B and H in MHz, rates in Mbit/s."""
import numpy as np

def gain(dx, h, pl0=128.1, slope=37.6, rng=None, fading=False):
    d_km = np.sqrt(np.asarray(dx, float) ** 2 + h ** 2) / 1000.0
    pl_db = pl0 + slope * np.log10(d_km)
    g = 10.0 ** (-pl_db / 10.0)
    if fading and rng is not None:
        g = g * rng.exponential(1.0, size=g.shape)   # Rayleigh power fading
    return g

def H_mhz(g, p=1.0, noise_dbm_hz=-174.0):
    """H = p g / sigma^2 expressed in MHz so that SNR = H / B (B in MHz)."""
    sigma2_w_hz = 10.0 ** ((noise_dbm_hz - 30.0) / 10.0)
    return p * g / sigma2_w_hz / 1e6

def rate_mbps(B_mhz, H):
    return B_mhz * np.log2(1.0 + H / B_mhz)
