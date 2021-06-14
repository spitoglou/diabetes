from src.helpers.diabetes.cega import clarke_error_grid


def test_zones():
    y_true = [50, 60, 100, 135, 150, 200, 250, 300]
    y_pred1 = [25, 45, 150, 160, 200, 300, 350, 390]
    y_pred2 = [75, 75, 50, 110, 100, 100, 150, 210]
    assert clarke_error_grid(y_true, y_pred1, "")[1] == [3, 5, 0, 0, 0]
    assert clarke_error_grid(y_true, y_pred2, "")[1] == [1, 4, 0, 3, 0]
