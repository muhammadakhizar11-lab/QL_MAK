# SPEC — Fragrance Cross-Venue Arbitrage (`frag-arb`)

Authoritative spec. Claude Code reads this first and treats it as the source of truth.
**Part A** says *what* is required. **Part B** says *how* it is built and in what order.

---

# PART A — REQUIREMENTS

## A1. Objective

Buy fragrance bottles that are listed below their exit value in UK peer-to-peer groups (Facebook, WhatsApp, r/UKFragranceSwap). Sell them across all venues.

Show, with point-in-time discipline and honest statistics, whether a fair-value signal identifies trades that are profitable after costs and that sell within 7 days.

This is a **portfolio project** to demonstrate quantitative trading skills. Success means a credible, reproducible answer, positive or negative, plus a small log of real trades.

## A2. Scope

| In scope | Out of scope |
|---|---|
| Bottles: new, used, part-full (**≥50% full** to buy), testers | Decants and samples; buying bottles under 50% full |
| Buying in groups (FB, WhatsApp, Reddit) | Buying on Vinted |
| Selling across all venues | Automated collection from Facebook or Vinted |
| Products on the watchlist (`watchlist.yaml`) | Products outside the watchlist, unless added to it |
| Free tooling by default: rules, fuzzy matching, OCR, pretrained image embeddings with a small trained fill-level classifier | Training any model from scratch; any paid API as a *required* step |
| Optional Anthropic API backend (off by default) | |
| Own modelling: fair value, signal, validation | Live trading at scale |

## A3. Venues and data requirements

| Venue | Role | Collection | Outcome evidence |
|---|---|---|---|
| Facebook groups | Main price source; buy and sell | **Manual** screenshots plus post URL | Weak: sometimes edited to "sold", often not |
| WhatsApp groups | Buy and sell | Built-in chat export with media | Follow-up messages ("sold", "pending") |
| r/UKFragranceSwap | Buy and sell; **[WTB] posts give buyer-side (bid) prices** | Official Reddit **Data API** via PRAW (app registered at reddit.com/prefs/apps) | **No "sold" flair exists.** Evidence is sometimes a comment stating the result; otherwise there is none (censored) |
| Vinted | **Exit only** | **Manual** screenshots of comparable listings | **Reliable:** marked "Sold" automatically once the transaction completes |

**Facts confirmed by Muhammad**
- **Listing format (groups):** every sale post has a description, a photo of the bottle, and a tag showing the **seller's name and the date**. The tag can be in any form: handwritten, on a screen, typed in Word or Notes.
- **Volume:** high on Facebook, with **heavy cross-posting** of the same bottle across groups.
- **Bundles:** sometimes posted without per-item prices. Sellers usually sell items individually too.
- **Sold marking:** on Vinted it is automatic and reliable. On the group venues it is sometimes edited to "sold" but often not, and recent posts are more likely to still be live.
- **Shipping:** in groups, Royal Mail T48 (**£3.65**) is usually *included in the ask*. On Vinted, InPost (**~£2.55**) is **paid by the seller**, and buyers pay the Vinted fee.
- **Vinted:** allows used or part-full fragrance if it is declared as such.
- **r/UKFragranceSwap posting conventions:**
  - The title must start with **[WTS]** (want to sell), **[WTT]** (want to trade), **[WTB]** (want to buy) or **[GIVEAWAY]**.
  - Every post must carry a flair: **Bottle**, **Decant** or **Bottle and Decant**.
  - **Use:** [WTS] posts with flair *Bottle* or *Bottle and Decant* are sell-side listings (keep only the bottle items). [WTB] posts are buy-side (bid) data. [WTT] posts are kept only where a cash price is stated. [GIVEAWAY] posts and *Decant*-only posts are excluded.
- **Versions:** where a fragrance comes in several versions (e.g. EDP and a pricier Extrait), each version is a **separate product with its own name** (e.g. "Baccarat Rouge 540 Extrait"). There is no separate concentration field.
- **Fill floor:** never buy a bottle that is **less than half full**.
- **Fast-sale window:** **7 days**. If a bottle hasn't sold by then, relist it.

## A4. Functional requirements

| ID | Requirement |
|---|---|
| F1 | Ingest all four venues into one `listings` table: UTC timestamps, GBP prices, one row per post. |
| F2 | Exclude decants and samples. |
| F3 | Extract structured fields per bottle: brand, fragrance (including the version in the name where one exists, e.g. "Sauvage Elixir"), size, fill, condition, batch, shipping status, price. If the version of a multi-version product can't be identified, flag the row. |
| F3b | **Fill floor:** bottles under 50% full are never flagged as buys. They stay in the data as price comparables for how fill level affects price. |
| F4 | Decode disguised or phonetic brand names using a growing alias table. |
| F5 | Determine fill level from photos when the text says "level as seen" or gives none. |
| F6 | **Deduplicate cross-posts:** the same bottle posted in several groups is one listing, not several. |
| F7 | Detect relists, including group → Vinted, using image embeddings plus field matching. The handwritten date tag also helps. |
| F8 | Normalise every price to **shipping-included**. The default is that shipping is included. If the post says to add T48, add postage, or offers "postage of your choice", **add £3.65**. |
| F9 | Split bundles when per-item prices are given. Exclude bundles without them from modelling, and count them. |
| F10 | Estimate a point-in-time fair value per bottle, with uncertainty (q10, q50, q90), for each exit venue. |
| F11 | Compute expected profit at the best exit and an uncertainty-scaled edge. Flag trades above the thresholds. |
| F12 | Measure whether edge predicts **sale within 7 days**, treating listings with no known outcome as **censored**, not unsold. |
| F13 | Run a paper portfolio and log 3–5 real round trips. |
| F14 | Produce a written research note with figures. |

## A5. Non-functional requirements

| ID | Requirement |
|---|---|
| N1 | **Point-in-time:** no feature, comparable or fair value may use information from after the listing's `posted_at`. This is enforced by tests. |
| N2 | **Privacy:** sellers' names, usernames and phone numbers are never stored in processed data (salted hashes only). **Photos show sellers' names on their tags**, so raw screenshots and photos stay on the local machine only. They are never added to Git or pushed to GitHub, and any image used in the write-up is redacted first. |
| N3 | **Compliance:** no automation against Facebook or Vinted. Reddit is accessed through the official API only. |
| N4 | **Reproducibility:** one command rebuilds the processed data from raw data. Any optional API calls are cached. All costs and thresholds live in `config.yaml`, including `extractor_backend` (`free` by default, or `api`). |
| N5 | **Honesty:** outcome definitions are fixed (A6) before results are seen. Results are reported whatever they show. |
| N6 | **Model error is measured:** every extraction component (rules, fuzzy matching, OCR, the fill classifier and the optional API backend) is scored against locked hand labels before use. |

## A6. Fixed definitions (pre-registered)

| Item | Definition |
|---|---|
| Product key | brand + fragrance name, with the version in the name where one exists (canonical) |
| Model target | **log(price ÷ (RRP per ml × size))**: price as a fraction of retail, shipping-included. Fill, condition and the other attributes are features, so sealed and used bottles of the same product price differently. May be revisited if a better target emerges; any change is logged. |
| Buy universe | Watchlist products, fill ≥50%, not collection-only, group venues only |
| Sold | Explicit evidence only: a "sold" edit, comment or message (groups); "Sold" status (Vinted). |
| Deleted post | **Primary:** censored at the last time it was seen live. **Sensitivity check:** treated as sold. |
| No evidence either way | Censored at the last time it was seen live (or at capture time). |
| Fast sale | Sold within **7 days** of `posted_at`. |
| Group exit cost | £3.65 (seller pays T48 from proceeds) |
| Vinted exit cost | £2.55 (seller pays InPost); £0 seller fee |
| Buy cost | The normalised ask, which already includes inbound shipping |
| Minimum profit | £10 (adjustable in `config.yaml`) |
| Live trade rule | List on the best exit; if not sold in 7 days, relist and log it. |

## A7. Known limitations (state these in the write-up)
- Outcome labels from the group venues are sparse and biased: sold edits are inconsistent, and recent posts are more often still live. Survival methods (censoring) handle this partly, not fully. Vinted's sold labels are reliable, so the exit side is cleaner than the entry side.
- Visible prices are asks. Negotiated final prices are rarely seen, so the ask-to-sale discount comes from a small sample.
- Vinted exit values come from a manual sample of comparables, not the full market.
- Manual capture is not random; record how listings were chosen.

---

# PART B — BUILD

## B1. Pipeline overview

```
 Facebook screenshots ─┐
 WhatsApp export ──────┤                                     ┌─► Fair value (group)  ─┐
 Reddit API ───────────┼─► 1 Ingest ─► 2 Extract text ─► 3 Fill level ─► 4 Dedupe/match ─► 5 Fair value ─┤                        ├─► 6 Signal ─► 7 Validate ─► 8 Report
 Vinted screenshots ───┘                                     └─► Vinted mark-up      ─┘
```

## B2. Stages

### Stage 1 — Ingest
- **Inputs:** raw files per venue.
  - Facebook capture log columns: `capture_id, group_code, post_url, posted_at, captured_at, screenshot_path, is_backfill`.
  - Facebook revisits log columns: `capture_id, checked_at, status, new_price, final_price_if_stated`.
- **Process:**
  - Parse the WhatsApp export (UK and US date formats), linking each image to its message.
  - Pull Reddit history plus new posts with PRAW, storing raw JSON snapshots. Route posts by title tag and flair, as set out in A3.
  - Apply the salted hash to every person identifier.
- **Output:** `listings` (listing_id, venue, group_code, posted_at, seller_hash, raw_text, image_paths, is_backfill).
- **Done when:** every venue loads; there is no real name in any processed file (a test scans for it).

### Stage 2 — Text extraction (free by default)
- **Process:**
  - **Screenshots:** run free OCR (Tesseract or PaddleOCR) to get the text; WhatsApp and Reddit already provide text.
  - **Product matching:** fuzzy and phonetic matching of the text against the watchlist, plus the alias table for disguised or phonetic spellings (e.g. `rapidfuzz`). Match on brand and fragrance together. Rows that don't match confidently go to a **review queue**; Muhammad resolves them, and each resolution adds an alias.
  - **Rules (regex):** price, size in ml, stated fill percentage, condition (sealed, boxed, tester), batch, and `shipping_status` ∈ {included, extra, collection}. "Add T48" or "postage of your choice" counts as `extra`.
  - Split bundles when per-item prices are given; flag unpriced bundles.
  - Filter out decants and samples by keyword rules.
  - Flag rows where the version of a multi-version product is unclear.
- **Optional backend:** when `extractor_backend: api`, the Anthropic API is used for rows the free path can't resolve (or for all rows). Responses are cached by input hash. The pipeline must produce a complete `extracted` table with the API switched off.
- **Output:** `extracted`. Columns:
  - product_key, size_ml
  - fill_bucket, fill_source, condition, batch_year, discontinued
  - ask_gbp_raw, shipping_status, ask_gbp (normalised), price_per_ml
  - is_bundle_unpriced, extract_confidence, match_method (rule / fuzzy / alias / manual / api)
- **Done when:** field accuracy and the share of rows sent to the review queue are reported against about 200 hand-labelled listings.

### Stage 3 — Fill level
- **Step 1 — text first:** where the post states a fill percentage ("90% full", "sealed"), use it (`fill_source = text`).
- **Step 2 — photo classifier:** where the text says "level as seen" or gives nothing, predict fill from the photo.
  - **Model:** pretrained image embeddings (CLIP or DINOv2, PyTorch, run on the Mac CPU) feeding a small trained head (logistic regression first; a small PyTorch head if that's weak). Fine-tune the backbone only if the head is clearly not enough.
  - **Target:** the ordered buckets {sealed, 95–100, 75–95, 50–75, <50}. The decision that matters most is **≥50% or not** (the fill floor), so report that split separately.
  - **Training labels:**
    - *Weak labels:* posts that state a fill percentage in text, paired with their photos.
    - *Hand labels:* a locked set of about 150 photos, **never used for training**, as the test set.
  - **Known bias:** posts that state their fill in text may differ from "level as seen" posts (different bottles, photos, sellers). The locked test set must be drawn from **"level as seen" posts**, so it measures the real use case.
  - **Optional comparison:** fill labels produced by Claude Code reading the photos in a session, saved to a labels file, can serve as a benchmark or fallback. This uses subscription allowance, not API billing.
- **Conflicts:** where text and photo disagree, record both and flag the row; never silently overwrite.
- **Vinted:** use the declared fill, with the classifier as a cross-check.
- **Done when:**
  - Bucket accuracy, within-one-bucket accuracy and the confusion matrix are reported on the locked test set.
  - Precision and recall of the ≥50% split are reported.
  - Error propagation (a one-bucket mistake) into fair value is measured (Part A, N6).
  - If the classifier is not good enough, the fallback is the rule: unknown fill → not a buy.

### Stage 4 — Dedupe and relist matching
- **Process:**
  - Reuse the image embeddings computed in Stage 3 (PyTorch, inference only).
  - Two listings are the **same item** when they share a seller hash and product key, their image similarity is above a threshold, and they fall within N days of each other.
  - **Cross-posts** (same seller, close in time) collapse into one listing, keeping the earliest `posted_at` and merging the outcome evidence.
  - **Relists** (later in time, or on another venue, possibly at a new price) are linked by `item_id` and give extra outcome evidence.
- **Output:** `items` (one row per physical bottle) and `item_events` (a timeline of posts and price changes per bottle).
- **Done when:** precision and recall are measured on about 100 hand-checked pairs, and the deduplicated counts are reported alongside the raw counts.

### Stage 5 — Fair value
- **Group model:**
  - LightGBM, quantile loss at q = 0.1, 0.5 and 0.9, on the target defined in A6.
  - **Trained on all prior listings** (not sold-only, since sold labels are too sparse), with listing age and outcome status as features.
  - **Features:**
    - size_ml, fill_bucket, condition, batch_year, discontinued
    - RRP per ml (hand-built table, one row per watchlist product)
    - brand
    - product_key (target-encoded out of fold only)
    - point-in-time rolling median of the target and comparable count for the product key
    - venue, month
- **Ask-to-sale adjustment:** estimate the discount between ask and sale price from items with a visible final price, and between unsold asks and sold asks. Apply it to q50 and widen the interval to reflect its uncertainty.
- **Sparse products:** where a product has fewer than k comparables, fall back on a brand-level estimate, widen the interval, and report coverage separately.
- **Vinted exit value:** group fair value × a Vinted mark-up, estimated from Vinted comparables by brand tier and shrunk towards the overall mark-up when the sample is thin. Report it with confidence intervals.
- **Validation and tuning:**
  - Expanding-window walk-forward by `posted_at`, with an embargo.
  - Optuna on the training folds only; the final time block is held out and touched once.
- **Done when:** pinball loss, 80% interval coverage (target close to 80%) and MAE are reported, with dense and sparse products shown separately.

### Stage 6 — Signal
- **Exit value per venue:** `exit_v = q50_v − exit_cost_v`.
- **Best exit:** `best_exit = max_v exit_v`.
- **Expected profit:** `exp_profit = best_exit − ask_gbp`.
- **Edge:** `exp_profit / (q90 − q10)` at the best-exit venue.
- **Trade flag:** `edge > θ`, `exp_profit > £10`, inside the buy universe (A6: watchlist, fill ≥50%, group venue, not collection-only). θ is chosen on validation data only.
- **Bid check (Reddit):** where an open [WTB] post exists for the same product, record the stated bid next to the flag as extra evidence of exit demand.

### Stage 7 — Validation
- **Primary test (survival):**
  - Kaplan–Meier curve of time to sale by edge quintile, with censoring as defined in A6.
  - Report P(sold ≤ 7 days) for each quintile, with confidence intervals.
  - A real signal shows a **monotone** relationship.
  - Confirm with a Cox model (edge plus controls).
- **Secondary tests:**
  - Relist price changes against edge.
  - The Vinted mark-up estimated on **sold** Vinted listings (reliable labels) against live ones, to check the gap is real and not just high unsold asks.
- **Sensitivity checks:**
  - Deleted posts treated as sold.
  - Haircut on the exit price.
  - Size of the Vinted mark-up.
  - Each cost parameter.
- **Paper portfolio:** buy every flagged listing at its ask and exit at the best-exit fair value minus a haircut. Report hit rate, P&L per trade, capital tied up and holding time, with bootstrap confidence intervals.
- **Live trades:** 3–5 real round trips. Log the model's prediction at entry, the entry price, the exit venue and price, costs, days to sell and any relists.
- **Robustness:** split results by venue, brand tier and time.

### Stage 8 — Report
- README plus a research note:
  - question, data, method, results, limitations (A7)
  - the live trade log
  - redacted figures

## B3. Repo layout
```
frag-arb/
  SPEC.md  CLAUDE.md  README.md  pyproject.toml  config.yaml  .env.example
  data/{raw,interim,processed,labels}/   # raw/ git-ignored, labels/ append-only
  src/fragarb/
    ingest/{facebook.py,whatsapp.py,reddit.py,vinted.py}
    extract/{ocr.py,fuzzy.py,rules.py,aliases.py,shipping.py,bundles.py,review_queue.py,api_optional.py}
    extract/fill/{weak_labels.py,embed.py,train.py,predict.py}
    match/{embeddings.py,dedupe.py}
    model/{features.py,fair_value.py,vinted_markup.py,tune.py}
    signal/{edge.py}
    validate/{survival.py,backtest.py,live_trades.py}
    privacy.py
  tests/            # hermetic fixtures; point-in-time and privacy tests mandatory
  notebooks/        # analysis only
  reports/{figures,note.md}
```

## B4. Tech stack
- Python 3.12, `uv`, `ruff`, `pytest`
- Polars, LightGBM, Optuna, `lifelines` (survival analysis)
- PyTorch with `open_clip` or `transformers` (image embeddings and the small fill-level head), scikit-learn
- `rapidfuzz` (matching), Tesseract or PaddleOCR (OCR)
- PRAW, matplotlib
- Anthropic SDK (**optional**, only if `extractor_backend: api`)

## B5. Build plan (3 weeks)

| Days | Build | Manual work (Muhammad, in parallel) | Gate |
|---|---|---|---|
| 1 | Repo, Git, `CLAUDE.md`, `config.yaml`, privacy module + tests | Register the Reddit app; get the r/UKFragranceSwap rules; start Facebook backfill capture | Repo runs; privacy test passes |
| 2–4 | Stage 1, all venues | Facebook backfill (aim for 300+); export WhatsApp; Vinted comparables for the top products | `listings` built |
| 4–5 | Hand-label set (text fields, fill test photos, dedupe pairs) | Labelling | Labels locked |
| 6–8 | Stage 2: OCR, fuzzy matching, rules, review queue; accuracy report | Build the RRP table; clear the review queue | Field accuracy reported |
| 8–10 | Stage 3: weak labels, embeddings, fill classifier; accuracy report | Label extra fill photos if the classifier needs them | Locked-test accuracy reported |
| 10–11 | Stage 4 dedupe and relist matching | Continue forward captures and revisits | Raw vs deduplicated counts reported |
| 12–14 | Stage 5 fair value, Optuna, Vinted mark-up | **Start live trades now**, so they finish within the 3 weeks | Coverage close to 80% |
| 15–17 | Stages 6–7 signal and survival validation | Log live trades | Pre-registered tests run |
| 18–21 | Stage 8 report; sensitivity and robustness checks | Close live trades | Note written |

Week 1 is mostly manual data collection, and that is expected.

## B6. Rules for Claude Code
- **Polars is used for all data handling** (no pandas, except where a library requires it at the boundary).
- **Optuna is used for all hyperparameter tuning.**
- Every feature builder takes an `as_of` timestamp and has a point-in-time test.
- All costs and thresholds are read from `config.yaml`.
- Work on feature branches, with small commits and clear messages.

---

# ASKS, NEEDS AND OPEN QUESTIONS

## Decisions needed
1. **Watchlist:** confirm the product list in chat; it is then saved as `watchlist.yaml`.

## From Muhammad
1. **Register a Reddit app for the Data API** at reddit.com/prefs/apps on day 1, since approval may take weeks.
   - *Not* the Devvit API (developers.reddit.com): Devvit apps only read the subreddit they are installed on, and installing needs a moderator.
2. **(Optional) Anthropic API key**, only if you switch `extractor_backend` to `api`. Put it in `.env`, never in Git. The project runs without it.
3. **Facebook backfill:** 300+ bottle listings for watchlist products (screenshot plus post URL plus posting date), from the groups with the most history.
4. **WhatsApp exports** with media, for each relevant group.
5. **Vinted comparables:** 3–5 per watchlist product, sold ones preferred.
6. **RRP table:** retail price and size for every watchlist product (each version is its own row).
7. **Hand labels:** ~200 listings (text fields), ~150 "level as seen" photos for the **locked fill-level test set**, ~100 pairs (same bottle or not). Also clear the match review queue as it fills, and label extra fill photos if the classifier needs them.
8. **Live trades:** budget for 3–5 bottles, and a log of each one.
9. **Courtesy check:** ask the Facebook group admins whether they mind the data being used.

## To find out during the build
- How far back the Reddit API reaches for r/UKFragranceSwap.
- What share of group posts show any evidence of being sold. If it's under ~10%, Reddit, the forward revisits and Vinted carry the validation.
- Bundle frequency (unpriced bundles are excluded and counted).
- **k** (the minimum number of comparables), dedupe and matching thresholds, and θ (the trade threshold). Set these from the data and log them.

