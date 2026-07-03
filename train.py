#!/usr/bin/env python3
"""Train a single-feature linear regression (price vs mileage) by gradient descent.

The subject's update rule diverges on raw mileage (~1e5), so we standardise the
feature for training and then fold the scaling back into theta0/theta1 so the
saved model is in raw units: estimatePrice(mileage) = theta0 + theta1 * mileage.
Only +, -, *, / are used (no polyfit / ML lib), per the subject.
"""
import csv
import sys

THETAS_FILE = 'thetas.csv'


def load_data(path):
    xs, ys = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            xs.append(float(row['km']))
            ys.append(float(row['price']))
    return xs, ys


def train(xs, ys, lr=0.1, iters=2000):
    m = len(xs)
    mean = sum(xs) / m
    std = (sum((x - mean) ** 2 for x in xs) / m) ** 0.5
    xn = [(x - mean) / std for x in xs]           # standardised feature

    t0 = t1 = 0.0
    for _ in range(iters):
        errs = [(t0 + t1 * xn[i]) - ys[i] for i in range(m)]
        g0 = sum(errs) / m
        g1 = sum(errs[i] * xn[i] for i in range(m)) / m
        t0, t1 = t0 - lr * g0, t1 - lr * g1        # simultaneous update

    # fold standardisation back into raw-scale thetas
    theta1 = t1 / std
    theta0 = t0 - t1 * mean / std
    return theta0, theta1


def save_thetas(theta0, theta1, path=THETAS_FILE):
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow([theta0, theta1])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'data.csv'
    xs, ys = load_data(path)
    theta0, theta1 = train(xs, ys)
    save_thetas(theta0, theta1)
    print(f'Trained on {len(xs)} rows -> theta0 = {theta0:.6f}, theta1 = {theta1:.6f}')
    print(f'Saved to {THETAS_FILE}')


if __name__ == '__main__':
    main()
