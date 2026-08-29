# ft_linear_regression

Linear regression by gradient descent, predicting car price from mileage. Pure
standard library: no machine learning package. A 42 Abu Dhabi project.

## Run

```
python3 train.py      # writes thetas.csv
python3 predict.py    # prompts for a mileage
```

`predict.py` returns 0 before training, as the subject requires.

## Test

```
python3 test_linear.py
```

Checks that the slope is negative, the intercept is near 8499, the fit beats a
zero model, and that saving and loading round-trips.

## Notes

- Training standardises mileage, then folds the scaling back so the saved thetas
  are in raw units.
- Converged fit: theta0 about 8499.60, theta1 about -0.021449.
- `PROGRESS.md` is the development log.
