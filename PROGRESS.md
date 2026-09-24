# ft_linear_regression — progress

**Status: mandatory DONE & verified.**

- `train.py` — gradient descent (subject's exact update rule) on standardised
  mileage, then folds scaling back so saved thetas are raw-scale. Writes `thetas.csv`.
- `predict.py` — prompts for mileage, prints `theta0 + theta1*mileage`. Defaults to
  0/0 when no `thetas.csv` (i.e. before training), as required.
- `test_linear.py` — trains on data.csv, checks slope<0, intercept≈8499, MSE beats
  zero model, save/load round-trip. **Run:** `python3 test_linear.py`

Converged fit: theta0 ≈ 8499.60, theta1 ≈ -0.021449 (canonical result).

**Run:** `python3 train.py` then `python3 predict.py`. Pure stdlib, no ML lib.

## Bonus - DONE
- `plot.py` — data + regression line (matplotlib), saves `plot.png`.
- `precision.py` — MSE, RMSE, MAE, R² (R² ≈ 0.733 on data.csv).

## Hardening
- `predict.py` rejects negative, nan, inf mileage; corrupt `thetas.csv` gives a clean error.
- `train.py` gives a clean error on missing file, missing column, bad value, <2 rows, equal mileages.
