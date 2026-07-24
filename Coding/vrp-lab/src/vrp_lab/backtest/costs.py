"""Transaction cost models. Costs are charged per fill, in dollars, always >= 0.

VX futures (per contract): half the quoted bid-ask spread (you cross it),
fixed commission, and linear slippage in size. SPY (per share): half spread
in bps of price plus per-share commission.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VXCostModel:
    """Cost of trading VX futures.

    half_spread_pts : half of the bid-ask, in VIX points (0.025 = 5-cent spread).
    commission : dollars per contract per side.
    slippage_pts_per_contract : linear impact, VIX points added per contract.
    multiplier : dollars per VIX point ($1000 for VX).
    """
    half_spread_pts: float = 0.025
    commission: float = 1.32
    slippage_pts_per_contract: float = 0.01
    multiplier: float = 1000.0

    def cost(self, qty: float) -> float:
        """Total dollar cost of trading ``qty`` contracts (sign ignored).

        Slippage is linear (0.01 pts on every contract) — appropriate for
        sizes that are noise next to VX front-month volume (~200k/day).
        A desk running hundreds of contracts per clip would need a proper
        square-root impact model; noted in METHODOLOGY.md.
        """
        n = abs(qty)
        per_contract_pts = self.half_spread_pts + self.slippage_pts_per_contract
        return per_contract_pts * self.multiplier * n + self.commission * n


@dataclass(frozen=True)
class EquityCostModel:
    """Cost of trading the equity underlying (delta hedge leg).

    half_spread_bps : half spread as basis points of trade price.
    commission_per_share : dollars per share.
    """
    half_spread_bps: float = 0.5
    commission_per_share: float = 0.0035

    def cost(self, qty: float, price: float) -> float:
        n = abs(qty)
        return n * price * self.half_spread_bps * 1e-4 + n * self.commission_per_share


@dataclass(frozen=True)
class OptionCostModel:
    """Options cost: cross a proportional half-spread on premium, floored.

    rel_half_spread : half spread as a fraction of option mid (options are
    wide — 2.5% default half-spread is optimistic for wings, said out loud
    in DATA_SOURCES.md).
    """
    rel_half_spread: float = 0.025
    min_per_contract: float = 0.05   # 5 cents/share floor
    commission: float = 0.65
    contract_size: float = 100.0

    def cost(self, qty: float, mid_price: float) -> float:
        n = abs(qty)
        half_spread = max(mid_price * self.rel_half_spread, self.min_per_contract)
        return n * (half_spread * self.contract_size + self.commission)
