#!/usr/bin/env python3
"""Bonus: report how well the trained thetas fit the dataset.

MSE/RMSE/MAE are in price units; R² is the share of price variance the line explains.
"""
import sys
import train
import predict


def metrics(xs, ys, theta0, theta1):
    m = len(ys)
    errs = [predict.estimate(x, theta0, theta1) - y for x, y in zip(xs, ys)]
    mse = sum(e * e for e in errs) / m
    mean_y = sum(ys) / m
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    return {
        'MSE': mse,
        'RMSE': mse ** 0.5,
        'MAE': sum(abs(e) for e in errs) / m,
        'R2': 1 - mse * m / ss_tot,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'data.csv'
    try:
        xs, ys = train.load_data(path)
    except (OSError, KeyError, ValueError, TypeError) as e:
        sys.exit(f'Error: cannot load {path}: {e}')
    if len(ys) < 2 or len(set(ys)) < 2:
        sys.exit(f'Error: {path} needs at least 2 different prices')
    theta0, theta1 = predict.load_thetas()
    if (theta0, theta1) == (0.0, 0.0):
        print('Warning: no trained thetas, run train.py first.')
    for name, value in metrics(xs, ys, theta0, theta1).items():
        print(f'{name:>4}: {value:.4f}')


if __name__ == '__main__':
    main()
