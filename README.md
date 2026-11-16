# NYC House Price Predictor

A machine learning project predicting New York City property prices using real listing data. I built four progressively richer models to measure **which features actually drive price** — and where linear regression breaks down.

## Dataset

**File:** `NY-House-Dataset.csv` (~4,800 raw listings, Kaggle "New York Housing Market")

I cleaned the dataset aggressively before modeling:
- **Removed 32-bit overflow:** a listing priced at `2147483647` (2³¹−1) is a data-entry error, not a $2.1B property.
- **Removed scraper placeholders:** `PROPERTYSQFT = 2184.207862` and `BATH = 2.3738608579684373` appeared on hundreds of unrelated listings. Dropped.
- **Deduplicated** identical listings.
- **Kept only residential types** (Condo, House, Co-op, Townhouse). Removed Land, Foreclosure, Pending — those are a different market with meaningless beds/baths/sqft.

**Final dataset:** 2,225 usable listings.
**Target:** `log(PRICE)` — modeling raw price lets a single $56M penthouse dominate the loss function. Log-space compresses the $60K–$56M range (~3,000×) into ~7 units.

## Approach — Feature Ablation

Instead of one model, I trained four to measure each feature group's marginal contribution:

| Model | Features | R² | MAE (log) | MAE ($) |
|---|---|---|---|---|
| M1: Area only | `PROPERTYSQFT` | 0.538 | 0.576 | $1,444,569 |
| M2: + Beds + Baths | M1 + `BEDS`, `BATH` | 0.608 | 0.535 | $1,357,392 |
| M3: + Geo | M2 + `LATITUDE`, `LONGITUDE` | 0.625 | 0.519 | $1,337,851 |
| M4: + Borough | M3 + one-hot `SUBLOCALITY` | **0.785** | **0.395** | **$1,039,614** |

### What this tells us

- **Square footage alone gets to R² 0.54.** Real estate is fundamentally about size.
- **Beds and baths add only ~7 points.** They're correlated with size, so they mostly repackage the same information.
- **Raw geo coordinates add only ~1.7 points.** A linear model over lat/lon can only fit one flat plane over the city — a poor representation of neighborhood effects.
- **Borough as a categorical flag adds ~16 points** — the biggest single jump. This is the key insight: **location matters as neighborhood identity, not as coordinates.** A Brooklyn co-op and a Manhattan co-op of the same size are different products.

## Interpretation

Coefficients for M3 (all on `log(price)` scale):

| Feature | Coefficient | Effect |
|---|---|---|
| `PROPERTYSQFT` | 0.0002 | +0.02% per sqft |
| `BEDS` | -0.0399 | ⚠️ see note below |
| `BATH` | 0.3615 | +44% per bathroom |
| `LATITUDE` | 1.4598 | ⚠️ see note below |
| `LONGITUDE` | -1.7925 | ⚠️ see note below |

**⚠️ Why `BEDS` is negative:** multicollinearity. `BEDS` is highly correlated with `PROPERTYSQFT` (bigger house → more bedrooms). Once size is in the model, `BEDS` measures "bedrooms per square foot" — i.e., **smaller rooms**, which is a cheaper construction signal. The sign flips, but the interpretation makes sense in context.

**⚠️ Why lat/lon coefficients are unreliable:** latitude and longitude are correlated with each other and with borough, so their individual coefficients are not stable across model specifications. The *combined* location signal is real (it drove the M3 fit), but the individual signs should not be over-interpreted.

## Honest limitations

- **MAE of ~$1M** reflects the wide price range in the data; log-MAE of 0.395 means predictions are typically within **~48% of true price** (e^0.395 ≈ 1.48×).
- **The model under-predicts luxury.** The residual plot (see `residuals.png`) shows variance exploding at $5M+: the model has seen few ultra-luxury listings and regresses them toward the mean. This is expected — luxury prices are driven by factors not in the dataset (views, building prestige, celebrity history), not by square footage.
- **Small townhouse sample:** only 181 of 2,225 listings are townhouses, so those predictions are less reliable.

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook House_pricing_Detector.ipynb
