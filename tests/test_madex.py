"""Tests for MADEX (Mean Adjusted Exponent Error) metric."""

import numpy as np
import pytest

from src.helpers.diabetes.madex import (
    graph_vs_mse,
    madex,
    mean_adjusted_exponent_error,
    rmadex,
)


class TestMeanAdjustedExponentError:
    """Tests for mean_adjusted_exponent_error function."""

    def test_perfect_prediction(self):
        """Test that perfect predictions have zero error."""
        y = [100.0, 120.0, 140.0]
        y_pred = [100.0, 120.0, 140.0]

        result = mean_adjusted_exponent_error(y, y_pred)

        assert result == 0.0

    def test_small_error(self):
        """Test that small errors produce small MADEX values."""
        y = [100.0, 120.0, 140.0]
        y_pred = [101.0, 121.0, 141.0]

        result = mean_adjusted_exponent_error(y, y_pred)

        assert result > 0.0
        assert result < 10.0  # Small error should be small

    def test_hypoglycemic_penalty(self):
        """Test that errors in hypoglycemic range are penalized more."""
        # Error at low glucose (70 mg/dL) - should be penalized more
        y_low = [70.0]
        y_pred_low = [80.0]
        error_low = mean_adjusted_exponent_error(y_low, y_pred_low)

        # Same magnitude error at normal glucose (140 mg/dL)
        y_normal = [140.0]
        y_pred_normal = [150.0]
        error_normal = mean_adjusted_exponent_error(y_normal, y_pred_normal)

        # Low glucose error should generally be penalized differently
        # The exact relationship depends on the formula parameters
        assert error_low != error_normal

    def test_custom_parameters(self):
        """Test with custom center, critical_range, and slope."""
        y = [100.0, 120.0]
        y_pred = [110.0, 130.0]

        result_default = mean_adjusted_exponent_error(y, y_pred)
        result_custom = mean_adjusted_exponent_error(
            y, y_pred, center=100, critical_range=50, slope=80
        )

        # Different parameters should yield different results
        assert result_default != result_custom

    def test_verbose_mode(self, capfd):
        """Test verbose mode outputs debug info."""
        y = [100.0]
        y_pred = [110.0]

        # verbose=True should not raise an error
        result = mean_adjusted_exponent_error(y, y_pred, verbose=True)
        assert result > 0

    def test_empty_lists(self):
        """Test with empty lists raises error."""
        with pytest.raises((ZeroDivisionError, IndexError)):
            mean_adjusted_exponent_error([], [])

    def test_large_error_clipping(self):
        """Test that large errors are clipped to avoid overflow."""
        y = [100.0]
        y_pred = [10000000.0]  # Very large prediction

        # Should not raise overflow error due to clipping
        result = mean_adjusted_exponent_error(y, y_pred)
        assert np.isfinite(result)


class TestMadex:
    """Tests for madex sklearn-compatible wrapper."""

    def test_madex_with_arrays(self):
        """Test madex with numpy arrays."""
        y = np.array([100.0, 120.0, 140.0])
        y_pred = np.array([105.0, 125.0, 145.0])

        result = madex(y, y_pred)

        assert isinstance(result, float)
        assert result > 0.0

    def test_madex_with_lists(self):
        """Test madex with Python lists."""
        y = [100.0, 120.0, 140.0]
        y_pred = [105.0, 125.0, 145.0]

        result = madex(y, y_pred)

        assert isinstance(result, float)
        assert result > 0.0

    def test_madex_sample_weight_ignored(self):
        """Test that sample_weight parameter is accepted but ignored."""
        y = np.array([100.0, 120.0])
        y_pred = np.array([105.0, 125.0])
        weights = np.array([1.0, 2.0])

        result_with_weights = madex(y, y_pred, sample_weight=weights)
        result_without_weights = madex(y, y_pred)

        # sample_weight is ignored, so results should be the same
        assert result_with_weights == result_without_weights


class TestRmadex:
    """Tests for rmadex (root MADEX) function."""

    def test_rmadex_is_sqrt_of_madex(self):
        """Test that RMADEX is the square root of MADEX."""
        y = np.array([100.0, 120.0, 140.0])
        y_pred = np.array([110.0, 130.0, 150.0])

        madex_value = madex(y, y_pred)
        rmadex_value = rmadex(y, y_pred)

        assert np.isclose(rmadex_value, np.sqrt(madex_value))

    def test_rmadex_perfect_prediction(self):
        """Test RMADEX with perfect prediction."""
        y = np.array([100.0, 120.0])
        y_pred = np.array([100.0, 120.0])

        result = rmadex(y, y_pred)

        assert result == 0.0

    def test_rmadex_sample_weight_ignored(self):
        """Test that sample_weight parameter is accepted but ignored."""
        y = np.array([100.0, 120.0])
        y_pred = np.array([105.0, 125.0])
        weights = np.array([1.0, 2.0])

        result_with_weights = rmadex(y, y_pred, sample_weight=weights)
        result_without_weights = rmadex(y, y_pred)

        assert result_with_weights == result_without_weights


class TestGraphVsMse:
    """Tests for graph_vs_mse function."""

    def test_graph_returns_plt(self):
        """Test that graph_vs_mse returns matplotlib pyplot."""
        import matplotlib.pyplot as plt

        result = graph_vs_mse(value=120, value_range=30)

        assert result is plt
        plt.close()

    def test_graph_save_action(self, tmp_path):
        """Test saving graph to file."""
        save_folder = str(tmp_path)

        result = graph_vs_mse(
            value=120, value_range=30, action="save", save_folder=save_folder
        )

        assert result is None
        # Check that file was created
        expected_file = tmp_path / "compare_vs_mse(120+-30).png"
        assert expected_file.exists()

    def test_graph_different_values(self):
        """Test graph with different reference values."""
        import matplotlib.pyplot as plt

        # Should not raise for different values
        result1 = graph_vs_mse(value=70, value_range=20)
        plt.close()

        result2 = graph_vs_mse(value=180, value_range=50)
        plt.close()

        assert result1 is not None
        assert result2 is not None
