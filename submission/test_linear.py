"""Self-check: train on data.csv and sanity-check the fit. Run: python3 test_linear.py"""
import os
import train
import predict
import precision


def mse(xs, ys, t0, t1):
    return sum((predict.estimate(x, t0, t1) - y) ** 2 for x, y in zip(xs, ys)) / len(xs)


if __name__ == '__main__':
    xs, ys = train.load_data('data.csv')
    t0, t1 = train.train(xs, ys)

    # price should fall with mileage, and the fit should beat the zero model
    assert t1 < 0, f'slope should be negative, got {t1}'
    assert 8000 < t0 < 9000, f'intercept out of expected range: {t0}'
    assert mse(xs, ys, t0, t1) < mse(xs, ys, 0.0, 0.0), 'trained model no better than zero'

    # predict before training (no thetas file) must yield 0
    assert predict.load_thetas('does_not_exist.csv') == (0.0, 0.0)

    # round-trip save/load
    train.save_thetas(t0, t1, 'thetas.csv')
    assert predict.load_thetas('thetas.csv') == (t0, t1)
    os.remove('thetas.csv')

    # bonus: R² on this dataset is ~0.733
    r2 = precision.metrics(xs, ys, t0, t1)['R2']
    assert 0.70 < r2 < 0.76, f'R2 out of expected range: {r2}'

    # bad training data must raise, not crash with ZeroDivisionError
    for bad_xs, bad_ys in (([1.0], [1.0]), ([5.0, 5.0], [1.0, 2.0])):
        try:
            train.train(bad_xs, bad_ys)
            raise AssertionError(f'train accepted bad data {bad_xs}')
        except ValueError:
            pass

    print(f'OK: fit theta0={t0:.4f} theta1={t1:.6f}, all checks pass')
