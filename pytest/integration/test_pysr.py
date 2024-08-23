def test_integration_pysr():
    "Simple PySR search"
    import pysr
    import numpy as np

    rng = np.random.RandomState(0)
    X = rng.randn(100, 5)
    y = np.cos(X[:, 0] * 2.1 - 0.5) + X[:, 1] * 0.7
    model = pysr.PySRRegressor(
        niterations=30,
        unary_operators=["cos"],
        binary_operators=["*", "+", "-"],
        early_stop_condition=1e-5,
    )
    model.fit(X, y)
    assert model.equations_.iloc[-1]["loss"] < 1e-5

