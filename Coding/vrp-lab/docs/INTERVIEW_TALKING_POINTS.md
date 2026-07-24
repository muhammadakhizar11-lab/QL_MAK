# Interview talking points

The questions this project should provoke, and the answers it earned.
Numbers below are from the actual runs in this repo (2013-05 → 2026-07,
free CBOE data).

## "Why does the VRP exist?"

Three reinforcing mechanisms, in rough order of size:

1. **Structural insurance demand.** Institutions are net buyers of index
   protection (puts, VIX calls) for mandate and regulatory reasons; they pay
   above actuarial value the way any insurance buyer does. The seller's
   compensation *is* the premium.
2. **Behavioral overestimation of tail probability** after every crisis —
   implied tails stay fat long after realized tails thin out.
3. **Dealer hedging flows.** Dealers short gamma from sold structures must
   buy vol when vol rises — flows amplify spikes, spikes justify the premium.

Measured here: ex-post 21d VRP mean **+3.6 vol pts**, positive **84%** of
days, but skew **−3.6** with a worst observation of **−69 pts** (Feb-2020
entries). It's an insurance book: steady coupon, occasional catastrophic claim.

## "What's the biggest risk?"

**Gap risk — the move that happens between your hedges.** Delta hedging
neutralizes direction continuously in theory, daily here; a gap goes straight
through it as $\tfrac12\Gamma (\Delta S)^2$ with no chance to re-hedge.
Feb 5 2018 is the canonical case: VIX from 17 to 37 in an afternoon, XIV
(−1x short-vol ETN) terminated at −96%. Our truly-naive replica **went
through zero that day** (worst day −112%, kurtosis 346). No daily-frequency
control catches a gap from a calm tape — that residual is why the position
cap and the tail overlay exist even though both cost return.

## "Walk me through your headline results — and why they're modest."

- Naive always-short front VX: **ruin in Feb 2018.** The point of the baseline.
- Regime-filtered (1x notional): survives everything, Sharpe ~0.14 — whipsaw
  (breaker covers near spike highs) plus ~23bp/side costs eat the carry.
- Filtered + vol-targeted (15% budget): max DD −23%, CVaR-99 −3.7%, zero
  margin breaches, but returns ≈ cash post-2013.
- Delta-hedged synthetic book: theta +$1.6M vs gamma −$1.1M on $1M equity —
  **theta+gamma > 0 is the premium, isolated** — but that's model-world
  (see caveat) and I'd say so in the first sentence.

I trust these numbers *because* they're modest: the free CBOE archive starts
mid-2013, so the sample excludes 2006–2012 where contango was fattest, and
every survival constraint is on. A backtest of this trade printing Sharpe 1.5
is usually reporting the premium gross of its claims.

## "What did you change after seeing results, and is that overfitting?"

Two changes, both mechanism-fixes rather than parameter tuning, both
documented in METHODOLOGY §6:

1. Exit hysteresis: exiting on any $VX1 < VIX$ blip covered 25 times in 15
   months in 2021 — each round trip sells the spike low. Threshold −2%.
2. Asymmetric gate: flattening at the VIX-25 bucket edge cost ~30%/yr in
   choppy years for the same reason. Now: calm to enter, genuine crisis
   (VIX>30 / deep inversion) to exit, acceleration halves exposure.

The honest answer on overfitting: these were diagnosed from *mechanism*
(trade logs showing systematic sell-low/buy-high), not from maximizing the
end number — and the end number is still unflattering, which is some
evidence I wasn't optimizing it.

## "How does this compare to a real vol desk?"

Honest gaps: daily frequency (a desk hedges intraday and gamma-scalps);
synthetic options with a *static* smile (no skew dynamics — a real short-put
book gets marked against steepening skew in selloffs); flat 30% margin vs
real SPAN with vega-netting; single underlying (a desk diversifies across
rates/FX/commodity vol where the VRP also lives); linear impact costs
(fine at 60 contracts, wrong at 600).

## "What would you do next with more data/infra?"

1. **OptionMetrics IvyDB** — replace the synthetic chains, get real skew
   dynamics, redo Phase 2 as an actual track record. Or pivot to **Deribit**:
   free, real historical chains; crypto VRP is well-documented.
2. **Longer-dated tenors** — VX4–VX7 carry has a fraction of the front's
   gamma; most of the post-2013 carry that survived costs lived there.
3. **Condition entries on the trailing VRP z-score** (built, not yet wired
   into the futures strategy) and on skew richness.
4. Pre-2013 VX archive to test the strategy where contango was fattest.
5. Intraday hedging and a square-root impact model.

## Design decisions worth defending

- **Event-driven engine, not vectorized** — vectorized backtests make
  look-ahead easy to write and impossible to see. The `MarketView` makes
  future data structurally unreachable, and a runtime guard raises if a
  window ever leaks.
- **Ex-post vs trailing VRP as two functions with different names** — the
  look-ahead distinction is a type-level decision, not a comment.
- **Vol targeting over premium-proportional sizing** — position shrinks when
  vol rises; the naive trade sizes up exactly when the tail fattens.
- **Report the whole distribution** (CVaR, worst day, skew, kurtosis,
  margin breaches) — a short-vol Sharpe without its left tail is marketing.
- **Ruin reports −100%, margin breaches are counted not hidden** — the
  naive baseline is kept *in* the results table on purpose.
