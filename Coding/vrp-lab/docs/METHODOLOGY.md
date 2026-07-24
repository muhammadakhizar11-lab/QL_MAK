# Methodology

Mathematical definitions and the modelling choices behind them. Anything
simplified is said out loud; the residual risk of each simplification is in
the final section.

## 1. Realized volatility estimators

All estimators return annualized vol over a rolling window of $n$ trading
days (default 21), annualization $\sqrt{252}$. Implementations:
`src/vrp_lab/volatility/realized.py`; recovery tests against simulated GBM
with known $\sigma$: `tests/test_realized.py`.

**Close-to-close** (baseline):

$$\hat\sigma^2 = \frac{252}{n-1}\sum_{i=1}^{n}(r_i-\bar r)^2,\qquad r_i=\ln\frac{C_i}{C_{i-1}}$$

**Parkinson (1980)** — high/low range; ~5x efficient vs close-to-close under
continuous GBM; biased low by discrete sampling; blind to overnight gaps:

$$\hat\sigma^2 = \frac{252}{4n\ln 2}\sum_{i=1}^{n}\left(\ln\frac{H_i}{L_i}\right)^2$$

**Garman–Klass (1980)** — OHLC; ~7.4x efficient; assumes zero drift and no
overnight jump:

$$\hat\sigma^2 = \frac{252}{n}\sum_{i=1}^{n}\left[\tfrac12\left(\ln\tfrac{H_i}{L_i}\right)^2-(2\ln2-1)\left(\ln\tfrac{C_i}{O_i}\right)^2\right]$$

**Rogers–Satchell (1991)** — drift-independent:

$$\hat\sigma^2 = \frac{252}{n}\sum_i\left[\ln\tfrac{H_i}{C_i}\ln\tfrac{H_i}{O_i}+\ln\tfrac{L_i}{C_i}\ln\tfrac{L_i}{O_i}\right]$$

**Yang–Zhang (2000)** — overnight + open-to-close + RS; minimum-variance
unbiased among OHLC estimators, handles gaps; **repo default for SPY**:

$$\hat\sigma^2_{YZ}=\sigma^2_{ON}+k\,\sigma^2_{OC}+(1-k)\,\sigma^2_{RS},\qquad k=\frac{0.34}{1.34+\frac{n+1}{n-1}}$$

with $\sigma^2_{ON}$ the sample variance of $\ln(O_i/C_{i-1})$ and
$\sigma^2_{OC}$ that of $\ln(C_i/O_i)$.

## 2. VRP definition — forward-looking vs trailing

$$VRP(t,\tau) = IV(t,\tau) - RV(t,\,t+\tau)$$

- **Ex-post** (`vrp_ex_post`): the RV leg is realized over the *future*
  window $(t, t+\tau]$. This is the true premium, unknowable at $t$.
  Measurement and research only; the last $\tau$ observations are NaN,
  never zero. Any appearance of this series in a signal path is a bug and
  tested against.
- **Trailing proxy** (`vrp_trailing`): the RV leg is $(t-\tau, t]$ —
  everything observable at $t$; the only variant a strategy may consume.
  Known bias: it lags at turning points, understating the premium right
  after a spike and overstating it right before one. The regime layer
  exists precisely because the proxy cannot carry that risk.

IV leg: the VIX, divided by 100. The VIX is the square root of the 30-day
SPX variance-swap rate — a model-free IV that sits slightly above ATM
Black-Scholes IV by smile convexity. Horizons: $\tau \in \{21, 63\}$ trading
days.

RV aggregation inside VRP uses the zero-mean convention
$RV=\sqrt{\tfrac{252}{\tau}\sum r_i^2}$ (drift is negligible at these
horizons and the variance-swap payoff is on squared returns).

## 3. VIX term structure

From per-contract CBOE settlements: VX1/VX2 = nearest/second not-yet-expired
contract (a contract leaves the curve on its expiry date). Front roll yield,
annualized by time-to-expiry:

$$\text{roll}_1 = \frac{VX1-VIX}{VIX}\cdot\frac{252}{dte_1}$$

Classification: contango ($VX2>VX1>VIX$), backwardation ($VX1<VIX$), else
mixed. Backwardation is the canonical crisis tell (Feb 2018, Mar 2020,
Aug 2024 all inverted).

VX expiry calendar: the Wednesday 30 days before the third Friday of the
following month; when that Friday is an exchange holiday, SPX expiration and
therefore the VX settlement move to the preceding business day (bites on
Good Friday Apr-2025 and Juneteenth Jun-2026 — validated against actual CBOE
contract files in `tests/test_term_structure.py`).

## 4. Backtest engine

Event-driven, one day per step, in strict order: (1) mark yesterday's book at
today's settle (variation margin + interest on cash), (2) strategy decision on
a `MarketView` structurally truncated at today, (3) fills at today's settle
plus costs, (4) end-of-day snapshot with SPAN-approximate margin check
(30% of gross notional).

No-look-ahead: strategies never see raw frames — the view slices `.loc[:t]`
and raises `LookAheadError` if a window ever extends past $t$. Fills at the
daily settle are a simplification; the cost model pays for the spread crossing
instead of modelling intraday fills.

Costs (per contract side): half-spread 0.025 VIX pts + linear slippage
0.01 pts + \$1.32 commission, ×\$1000 multiplier ⇒ ~\$36/contract, ~23bp of
notional at VIX 15. Linear impact is appropriate for sizes that are noise
next to ~200k contracts/day front-month volume; a desk-size book needs a
square-root impact model.

## 5. Sizing

Vol targeting: contracts $n$ such that position $\$$-vol matches a 15% annual
budget,

$$n=\frac{\sigma_{tgt}\,E}{\sigma_{instr}\sqrt{252}\cdot P\cdot m}\quad\text{capped at}\quad \frac{2E}{Pm},$$

with $\sigma_{instr}$ the trailing 21d daily vol of the front settle series.
The deliberate inversion vs naive intuition: position **shrinks** when vol
rises, exactly when the left tail fattens. Kelly is implemented
(`kelly_fraction`) and hard-capped at 0.25 — full Kelly on estimated moments
is a documented way to blow up.

## 6. Regime gate — asymmetric by design

Entries require calm: VIX not in the Crisis bucket (>25) **and** vol not
accelerating (5d RV / 21d RV ≤ 1.5). Exits require genuine crisis: VIX > 30
circuit breaker, backwardation deeper than −2%, or max holding period.
While vol is accelerating, held exposure is halved.

Why asymmetric: flattening whenever the VIX crossed 25 systematically covered
at spike highs and re-entered lower — measured at roughly −30%/yr in choppy
years (2021) during development. The −2% backwardation threshold is
hysteresis for the same reason: a −0.1% blip is noise, and exiting on it
sells the spike low. Both numbers live in `config/settings.yaml`.

## 7. Greeks P&L decomposition

Daily Taylor expansion around the prior close, greeks known at the prior
close (point-in-time):

$$\Delta V \approx \underbrace{\Delta\,dS}_{delta} + \underbrace{\tfrac12\Gamma\,dS^2}_{gamma} + \underbrace{\nu\,d\sigma}_{vega} + \underbrace{\Theta\,dt}_{theta} + \text{residual}$$

For the delta-hedged short book the delta term nets against the hedge, so the
economics reduce to **theta collected vs gamma paid — and
$E[\Theta dt + \tfrac12\Gamma dS^2] > 0$ over time is the harvested VRP.**
Tests assert the components sum to actual P&L with residual < 15% of gross.

Black–Scholes–Merton greeks (continuous $r$, dividend yield $q$):
$\Delta_c=e^{-qT}N(d_1)$, $\Gamma=\frac{e^{-qT}\phi(d_1)}{S\sigma\sqrt T}$,
$\nu=Se^{-qT}\phi(d_1)\sqrt T$, $\Theta$ standard calendar decay, with
$d_1=\frac{\ln(S/K)+(r-q+\sigma^2/2)T}{\sigma\sqrt T}$.

## 8. Synthetic options model (Phase 2) — the big caveat

No free historical SPY chains exist, so Phase 2 prices options with BSM from
real SPY closes, the real VIX as ATM IV, and a **fixed** parametric smirk
$\sigma(m)=\sigma_{ATM}(1-0.15m+0.05m^2)$, $m=\ln(K/S)$.

What this buys: real underlying paths and real IV *levels*, so the theta/gamma
economics are driven by actual history. What it cannot capture, in order of
severity:

1. **No smile dynamics** — skew neither steepens in selloffs nor bleeds in
   rallies; short-put books look better than they are.
2. **Flat term structure** — 30-day VIX applied to every tenor.
3. **Variance-swap vs ATM basis** — VIX overstates ATM IV by convexity;
   inflates collected premium slightly.
4. **No microstructure** — mid pricing plus a proportional cost model;
   real wings trade far wider.

Every Phase-2 number carries this asterisk. The delta-hedged Sharpe is a
*sketch of the mechanics*, not a track record.

## 9. Known limitations (beyond §8)

- Daily frequency: no intraday hedging, no gap modelling beyond close-to-close.
- Margin is a flat 30% of notional, not real SPAN; no margin-call liquidation
  mechanics (breaches are recorded and asserted instead).
- CBOE's free VX archive starts mid-2013: the 2006–2012 sample (fattest
  contango era, plus 2008) is out of reach; results are the *hard* subsample.
- Single underlying; no cross-asset vol.
- Trailing-vol sizing on the VX1 series includes roll-day jumps, mildly
  over-estimating vol and under-sizing (conservative direction).
