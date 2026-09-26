#!/usr/bin/env python3
"""Predict a car price from mileage using the trained thetas.

Before training (no thetas file), theta0 and theta1 are 0, so it predicts 0.
"""
import csv
import math
import sys

THETAS_FILE = 'thetas.csv'


def load_thetas(path=THETAS_FILE):
    try:
        with open(path) as f:
            t0, t1 = next(csv.reader(f))
            return float(t0), float(t1)
    except (FileNotFoundError, StopIteration):
        return 0.0, 0.0
    except ValueError:
        sys.exit(f'Error: {path} is corrupt. Run train.py again.')


def estimate(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


def main():
    theta0, theta1 = load_thetas()
    try:
        mileage = float(input('Enter a mileage (km): '))
    except (ValueError, EOFError):
        mileage = -1.0
    if not math.isfinite(mileage) or mileage < 0:
        print('Please enter a valid mileage (a number >= 0).')
        sys.exit(1)
    price = estimate(mileage, theta0, theta1)
    print(f'Estimated price: {max(price, 0):.2f}')


if __name__ == '__main__':
    main()
