#!/usr/bin/env python3
"""Bonus: plot the dataset and the trained regression line. Needs matplotlib."""
import sys
import train
import predict

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit('Error: matplotlib is missing. Run: pip install matplotlib')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'data.csv'
    try:
        xs, ys = train.load_data(path)
    except (OSError, KeyError, ValueError, TypeError) as e:
        sys.exit(f'Error: cannot load {path}: {e}')
    if not xs:
        sys.exit(f'Error: {path} has no rows')
    theta0, theta1 = predict.load_thetas()

    x_line = [0, max(xs)]
    plt.scatter(xs, ys, label='data')
    plt.plot(x_line, [predict.estimate(x, theta0, theta1) for x in x_line],
             color='red', label=f'price = {theta0:.2f} + ({theta1:.6f}) * km')
    plt.xlabel('mileage (km)')
    plt.ylabel('price')
    plt.title('ft_linear_regression')
    plt.legend()
    plt.savefig('plot.png')
    print('Saved to plot.png')
    plt.show()


if __name__ == '__main__':
    main()
