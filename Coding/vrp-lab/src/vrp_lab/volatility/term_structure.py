"""VIX / VX futures term structure: curve building, contango/backwardation, roll yield.

Input is the long-format VX frame from ``data.cboe.pull_vx_curve``
(index=trade date, columns [settle, volume, open_interest, expiry]).
Everything here is point-in-time by construction: on each trade date we only
use contracts observed (settled) on that date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import get_logger

log = get_logger(__name__)

TRADING_DAYS = 252


def build_curve(vx_long: pd.DataFrame, vix_close: pd.Series, n_tenors: int = 2) -> pd.DataFrame:
    """Pivot the long VX frame into a daily curve: VX1..VXn settles + days to expiry.

    VX1 = nearest not-yet-expired contract on each trade date (a contract is
    dropped from its own expiry date onward — on expiry day it settles to
    cash SOQ and is not a tradeable leg for our purposes).

    Returns columns: vix, vx1, vx2, ..., dte1, dte2, ..., expiry1, expiry2, ...
    """
    rows = []
    for date, grp in vx_long.groupby(level=0):
        live = grp[grp["expiry"] > date].sort_values("expiry")
        row: dict = {"date": date}
        for i in range(n_tenors):
            if i < len(live):
                rec = live.iloc[i]
                row[f"vx{i+1}"] = float(rec["settle"])
                row[f"dte{i+1}"] = int((rec["expiry"] - date).days)
                row[f"expiry{i+1}"] = rec["expiry"]
        rows.append(row)
    curve = pd.DataFrame(rows).set_index("date").sort_index()
    curve["vix"] = vix_close.reindex(curve.index)
    missing = curve["vx1"].isna().sum() + curve["vix"].isna().sum()
    if missing:
        log.warning("term structure: %d missing vx1/vix values after alignment", int(missing))
    return curve


def classify_regime(curve: pd.DataFrame) -> pd.Series:
    """Label each day 'contango' (VX2 > VX1 > VIX), 'backwardation' (VX1 < VIX),
    or 'mixed' (anything else, e.g. front in contango but flat belly)."""
    contango = (curve["vx2"] > curve["vx1"]) & (curve["vx1"] > curve["vix"])
    backward = curve["vx1"] < curve["vix"]
    out = pd.Series("mixed", index=curve.index, name="ts_regime")
    out[contango] = "contango"
    out[backward] = "backwardation"
    return out


def roll_yield(curve: pd.DataFrame) -> pd.DataFrame:
    r"""Annualized roll yields along the curve.

    front: :math:`\frac{VX1 - VIX}{VIX}\cdot\frac{252}{dte_1}` — the carry a
    short front-month position earns if the curve is static and VX1 converges
    to spot VIX at expiry.
    second: :math:`\frac{VX2 - VX1}{VX1}\cdot\frac{252}{dte_2 - dte_1}`.

    Positive = contango (short vol collects carry); negative = backwardation.
    """
    front = (curve["vx1"] - curve["vix"]) / curve["vix"] * (TRADING_DAYS / curve["dte1"])
    gap = (curve["dte2"] - curve["dte1"]).replace(0, np.nan)
    second = (curve["vx2"] - curve["vx1"]) / curve["vx1"] * (TRADING_DAYS / gap)
    return pd.DataFrame({"roll_front_ann": front, "roll_second_ann": second})


def slope(curve: pd.DataFrame) -> pd.Series:
    """Simple front slope (VX1 - VIX)/VIX, un-annualized — the raw contango
    depth the entry threshold is quoted against."""
    return ((curve["vx1"] - curve["vix"]) / curve["vix"]).rename("ts_slope")


def inversion_flag(curve: pd.DataFrame) -> pd.Series:
    """True when the curve is inverted (VX1 < VIX) — the canonical crisis tell.

    Historically True around 2018-02-05..09, most of Mar 2020, and 2024-08-05.
    """
    return (curve["vx1"] < curve["vix"]).rename("inverted")
