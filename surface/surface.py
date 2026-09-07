"""
The queryable surface object: given (K, T), return IV, price, or greeks --
independent of how the surface was built. See BUILD_GUIDE.md section 5.

Interpolation across expiries: linear in TOTAL VARIANCE space (not IV
directly -- variance is the additive quantity over time). Flat-extrapolate
beyond the longest fitted expiry.
"""


class VolSurface:
    def __init__(self, svi_params_by_expiry: dict, forward_by_expiry: dict):
        """
        svi_params_by_expiry: {T: {'a':, 'b':, 'rho':, 'm':, 'sigma':}}
        forward_by_expiry: {T: forward_price}
        """
        self.svi_params_by_expiry = svi_params_by_expiry
        self.forward_by_expiry = forward_by_expiry

    def iv(self, K: float, T: float) -> float:
        """Implied vol at (K, T), interpolating in total variance space
        if T doesn't match a fitted expiry exactly."""
        raise NotImplementedError

    def price(self, K: float, T: float, option_type: str) -> float:
        raise NotImplementedError

    def greeks(self, K: float, T: float, option_type: str) -> dict:
        raise NotImplementedError
