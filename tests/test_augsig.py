import numpy as np
import pytest
from augsig import augment, Augment
from augsig.noisifier import noisify, burstify, burst_mask
from augsig.warper import (
    rand_knots,
    twarp_bezier, twarp_pchip,
    adrift_bezier, adrift_pchip,
    amod_bezier, amod_pchip,
    drift_linear,
)
from augsig.utils import normalize, freqfilt


# Shared fixtures
N = 500
RNG = np.random.default_rng(0)


@pytest.fixture
def signal():
    rng = np.random.default_rng(42)
    return rng.standard_normal(N)


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_output_range(self, signal):
        out = normalize(signal)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5

    def test_shape_preserved(self, signal):
        assert normalize(signal).shape == signal.shape

    def test_constant_signal_no_nan(self):
        out = normalize(np.ones(100))
        assert not np.any(np.isnan(out))


class TestFreqfilt:
    def test_lowpass_attenuates_high_freq(self, signal):
        filtered = freqfilt(signal, 0.0, 0.1)
        assert np.std(filtered) < np.std(signal)

    def test_passthrough_full_band(self, signal):
        out = freqfilt(signal, 0.0, 1.0)
        np.testing.assert_array_equal(out, signal)

    def test_shape_preserved(self, signal):
        assert freqfilt(signal, 0.0, 0.5).shape == signal.shape

    def test_invalid_cutoffs_raises(self, signal):
        with pytest.raises(ValueError):
            freqfilt(signal, 0.8, 0.2)

    def test_out_of_range_raises(self, signal):
        with pytest.raises(ValueError):
            freqfilt(signal, -0.1, 0.5)


# ---------------------------------------------------------------------------
# noisifier
# ---------------------------------------------------------------------------

class TestNoisify:
    def test_output_shape(self, signal):
        noisy, noise = noisify(signal, snr_db=20, rng=np.random.default_rng(0))
        assert noisy.shape == signal.shape
        assert noise.shape == signal.shape

    def test_inf_snr_returns_original(self, signal):
        noisy, noise = noisify(signal, snr_db=float('inf'))
        np.testing.assert_array_equal(noisy, signal)
        assert np.all(noise == 0)

    def test_snr_approximately_correct(self, signal):
        snr_db = 20
        noisy, noise = noisify(signal, snr_db=snr_db, rng=np.random.default_rng(1))
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        measured_snr_db = 10 * np.log10(signal_power / noise_power)
        assert abs(measured_snr_db - snr_db) < 2.0

    @pytest.mark.parametrize("color", ["white", "pink", "brown", "blue", "violet"])
    def test_noise_colors(self, signal, color):
        noisy, noise = noisify(signal, snr_db=15, color=color, rng=np.random.default_rng(2))
        assert noisy.shape == signal.shape
        assert not np.any(np.isnan(noise))

    @pytest.mark.parametrize("dist", ["gauss", "uniform", "laplace"])
    def test_noise_distributions(self, signal, dist):
        noisy, noise = noisify(signal, snr_db=15, dist=dist, rng=np.random.default_rng(3))
        assert noisy.shape == signal.shape

    def test_resample_dist(self, signal):
        noisy, noise = noisify(signal, snr_db=15, dist="resample",
                               resample_pool="self", rng=np.random.default_rng(4))
        assert noisy.shape == signal.shape

    def test_minus_inf_snr_unit_power(self, signal):
        noise, _ = noisify(signal, snr_db=float('-inf'), rng=np.random.default_rng(5))
        assert abs(np.mean(noise ** 2) - 1.0) < 0.1

    def test_reproducibility(self, signal):
        n1, _ = noisify(signal, snr_db=20, rng=np.random.default_rng(99))
        n2, _ = noisify(signal, snr_db=20, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(n1, n2)


class TestBurstify:
    def test_output_shape(self, signal):
        noisy, burst = burstify(signal, snr_db=float('-inf'), rng=np.random.default_rng(0))
        assert noisy.shape == signal.shape
        assert burst.shape == signal.shape

    def test_inf_snr_returns_original(self, signal):
        noisy, burst = burstify(signal, snr_db=float('inf'))
        np.testing.assert_array_equal(noisy, signal)
        assert np.all(burst == 0)

    def test_burst_is_sparse(self, signal):
        _, burst = burstify(signal, snr_db=float('-inf'), n_bursts=3,
                            burst_width=10, rng=np.random.default_rng(7))
        nonzero_ratio = np.count_nonzero(burst) / len(burst)
        assert nonzero_ratio < 0.5


# ---------------------------------------------------------------------------
# warper
# ---------------------------------------------------------------------------

class TestTimeWarp:
    @pytest.mark.parametrize("fn", [twarp_bezier, twarp_pchip])
    def test_output_shape(self, signal, fn):
        out = fn(signal, rng=np.random.default_rng(0))
        assert out.shape == signal.shape

    @pytest.mark.parametrize("fn", [twarp_bezier, twarp_pchip])
    def test_value_range_preserved(self, signal, fn):
        out = fn(signal, rng=np.random.default_rng(0))
        assert out.min() >= signal.min() - 1e-6
        assert out.max() <= signal.max() + 1e-6


class TestAmpDrift:
    @pytest.mark.parametrize("fn", [adrift_bezier, adrift_pchip])
    def test_output_shape(self, signal, fn):
        out = fn(signal, variance=0.05, rng=np.random.default_rng(0))
        assert out.shape == signal.shape

    @pytest.mark.parametrize("fn", [adrift_bezier, adrift_pchip])
    def test_small_variance_stays_close(self, signal, fn):
        out = fn(signal, variance=0.01, rng=np.random.default_rng(0))
        assert np.max(np.abs(out - signal)) < 0.1


class TestAmpMod:
    @pytest.mark.parametrize("fn", [amod_bezier, amod_pchip])
    def test_output_shape(self, signal, fn):
        out = fn(signal, variance=0.05, rng=np.random.default_rng(0))
        assert out.shape == signal.shape

    @pytest.mark.parametrize("fn", [amod_bezier, amod_pchip])
    def test_small_variance_stays_close(self, signal, fn):
        out = fn(signal, variance=0.01, rng=np.random.default_rng(0))
        np.testing.assert_allclose(out, signal, atol=0.1)


class TestDriftLinear:
    def test_output_shape(self, signal):
        out = drift_linear(signal, a=0.1, b=0.0, rng=np.random.default_rng(0))
        assert out.shape == signal.shape

    def test_zero_drift_unchanged(self, signal):
        out = drift_linear(signal, a=0.0, b=0.0, rng=np.random.default_rng(0))
        np.testing.assert_array_almost_equal(out, signal)

    def test_interval_sampling(self, signal):
        out = drift_linear(signal, a=[-0.5, 0.5], b=[-0.1, 0.1], rng=np.random.default_rng(0))
        assert out.shape == signal.shape

    def test_invalid_a_raises(self, signal):
        with pytest.raises(ValueError):
            drift_linear(signal, a=[1, 2, 3], b=0.0)


# ---------------------------------------------------------------------------
# augmenter
# ---------------------------------------------------------------------------

class TestAugment:
    def test_output_shape_single_aug(self, signal):
        config = {"a1": {"Add_noise": True, "SNRdb": 20}}
        out = augment(signal, config, seed=0)
        assert out.shape == (N, 2)  # original + 1 variant

    def test_original_preserved_in_col0(self, signal):
        config = {"a1": {"Add_noise": True, "SNRdb": 20}}
        out = augment(signal, config, seed=0)
        np.testing.assert_array_equal(out[:, 0], signal)

    def test_num_copies(self, signal):
        config = {"a1": {"num_copies": 3, "Add_noise": True, "SNRdb": 20}}
        out = augment(signal, config, seed=0)
        assert out.shape == (N, 4)  # original + 3 copies

    def test_multiple_recipes(self, signal):
        config = {
            "a1": {"Flip": True},
            "a2": {"Invert": True},
            "a3": {"Add_noise": True, "SNRdb": 15},
        }
        out = augment(signal, config, seed=0)
        assert out.shape == (N, 4)

    def test_normalize_output_true(self, signal):
        config = {"a1": {"Add_noise": True, "SNRdb": 20}}
        out = augment(signal, config, seed=0, normalize_output=True)
        variant = out[:, 1]
        assert variant.min() >= 0.0
        assert variant.max() <= 1.0 + 1e-5

    def test_normalize_output_false(self, signal):
        config = {"a1": {"Add_noise": True, "SNRdb": 5}}
        out_norm = augment(signal, config, seed=0, normalize_output=True)
        out_raw = augment(signal, config, seed=0, normalize_output=False)
        assert not np.allclose(out_norm[:, 1], out_raw[:, 1])

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            augment(np.ones((10, 2)), {})

    def test_reproducibility(self, signal):
        config = {"a1": {"Add_noise": True, "SNRdb": 20}}
        out1 = augment(signal, config, seed=42)
        out2 = augment(signal, config, seed=42)
        np.testing.assert_array_equal(out1, out2)

    def test_flip(self, signal):
        config = {"a1": {"Flip": True}}
        out = augment(signal, config, seed=0, normalize_output=False)
        np.testing.assert_array_equal(out[:, 1], np.flip(signal))

    def test_invert(self, signal):
        config = {"a1": {"Invert": True}}
        out = augment(signal, config, seed=0, normalize_output=False)
        np.testing.assert_array_almost_equal(out[:, 1], np.max(signal) - signal)

    @pytest.mark.parametrize("key,extra", [
        ("Bezier_time_warp", {}),
        ("PCHIP_time_warp", {}),
        ("Bezier_amp_drift", {}),
        ("PCHIP_amp_drift", {}),
        ("Bezier_amp_mod", {}),
        ("PCHIP_amp_mod", {}),
        ("lf_noise", {"lf_noise_amplitude": 0.05, "lf_noise_frequency": 0.05}),
        ("powerline", {"powerline_amplitude": 0.05, "powerline_frequency": 0.1}),
        ("burst", {"burst_amplitude": 0.1, "burst_number": 3}),
        ("drift", {"drift_a": [-0.1, 0.1], "drift_b": [-0.05, 0.05]}),
    ])
    def test_each_augmentation_runs(self, signal, key, extra):
        config = {"a1": {key: True, **extra}}
        out = augment(signal, config, seed=0)
        assert out.shape == (N, 2)
        assert not np.any(np.isnan(out))


class TestAugmentClass:
    def test_callable(self, signal):
        config = {"a1": {"Add_noise": True, "SNRdb": 20}}
        aug = Augment(config, seed=0)
        out = aug(signal)
        assert out.shape == (N, 2)

    def test_seed_override(self, signal):
        config = {"a1": {"Add_noise": True, "SNRdb": 20}}
        aug = Augment(config, seed=0)
        out1 = aug(signal, seed=99)
        out2 = aug(signal, seed=99)
        np.testing.assert_array_equal(out1, out2)

    def test_normalize_output_false(self, signal):
        config = {"a1": {"Add_noise": True, "SNRdb": 5}}
        aug = Augment(config, seed=0, normalize_output=False)
        out = aug(signal)
        assert out.shape == (N, 2)


# ---------------------------------------------------------------------------
# Regression tests for PhysioNet review comments (submission vuCy1q8QeG9EZqJLvMB1)
#
# Each class below pins one reviewer item so a later change cannot silently
# reintroduce the defect. Tags match PhysioNet_materials/reviewers-comments-s1.txt
# ---------------------------------------------------------------------------

WARP_FNS = [twarp_bezier, twarp_pchip, adrift_bezier,
            adrift_pchip, amod_bezier, amod_pchip]
RAND_KNOTS_FNS = [twarp_bezier, twarp_pchip, adrift_pchip, amod_pchip]
BEZIER_AMP_FNS = [adrift_bezier, amod_bezier]


class TestSignalShapeHandling:
    """B3c: bare squeeze() replaced by explicit 1D validation."""

    @pytest.mark.parametrize("fn", WARP_FNS)
    @pytest.mark.parametrize("shape", [(N,), (N, 1), (1, N)])
    def test_singleton_axes_accepted(self, fn, shape):
        x = np.random.default_rng(0).standard_normal(shape)
        out = fn(x, k=4, variance=0.01, rng=np.random.default_rng(0))
        assert out.shape == (N,)

    @pytest.mark.parametrize("fn", WARP_FNS)
    def test_true_2d_rejected(self, fn):
        x = np.zeros((3, 100))
        with pytest.raises(ValueError, match="1D array"):
            fn(x, k=4, variance=0.01, rng=np.random.default_rng(0))

    def test_error_reports_original_shape(self):
        with pytest.raises(ValueError, match=r"\(1, 2, 100\)"):
            twarp_bezier(np.zeros((1, 2, 100)), k=4, rng=np.random.default_rng(0))

    @pytest.mark.parametrize("shape", [(N,), (N, 1), (1, N)])
    def test_drift_linear_singleton_axes(self, shape):
        x = np.random.default_rng(0).standard_normal(shape)
        out = drift_linear(x, a=[-0.3, 0.3], b=[-0.1, 0.1], rng=np.random.default_rng(0))
        assert out.shape == (N,)

    def test_burst_mask_singleton_axes(self):
        noise = np.random.default_rng(0).standard_normal((1, N))
        out = burst_mask(noise, n_bursts=2, burst_width=10, burst_base=0.0, burst_onset=5)
        assert out.shape == (N,)

    def test_burst_mask_rejects_true_2d(self):
        with pytest.raises(ValueError, match="1D array"):
            burst_mask(np.zeros((3, 100)), n_bursts=1, burst_width=5,
                       burst_base=0.0, burst_onset=0)


class TestAugmentShapeHandling:
    """B3c / A2: augment() agrees with the functions it calls."""

    @pytest.mark.parametrize("shape", [(N,), (N, 1), (1, N)])
    def test_singleton_axes_accepted(self, shape):
        x = np.random.default_rng(0).standard_normal(shape)
        out = augment(x, {"a": {"Add_noise": True, "SNRdb": 20}}, seed=0)
        assert out.shape == (N, 2)

    def test_true_2d_rejected(self):
        with pytest.raises(ValueError, match=r"\(3, 100\)"):
            augment(np.zeros((3, 100)), {"a": {"Flip": True}}, seed=0)

    def test_list_input_accepted(self):
        x = list(np.random.default_rng(0).standard_normal(N))
        out = augment(x, {"a": {"Flip": True}}, seed=0)
        assert out.shape == (N, 2)


class TestKValidation:
    """B3a / B3b: k is validated, with the bound each group actually needs."""

    def test_rand_knots_returns_k_total_knots(self):
        for k in (3, 4, 5, 6):
            x_vals, y_vals = rand_knots(k=k, variance=1e-9, rng=np.random.default_rng(0))
            assert len(x_vals) == k
            assert len(y_vals) == k

    @pytest.mark.parametrize("fn", RAND_KNOTS_FNS)
    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_rand_knots_group_requires_k_ge_3(self, signal, fn, k):
        with pytest.raises(ValueError, match="k must be"):
            fn(signal, k=k, variance=0.01, rng=np.random.default_rng(0))

    @pytest.mark.parametrize("fn", BEZIER_AMP_FNS)
    def test_bezier_amp_rejects_k_zero(self, signal, fn):
        # k=0 produced an empty envelope: amod_bezier zeroed the signal outright
        with pytest.raises(ValueError, match="k must be"):
            fn(signal, k=0, variance=0.05, rng=np.random.default_rng(0))

    @pytest.mark.parametrize("fn", BEZIER_AMP_FNS)
    @pytest.mark.parametrize("k", [1, 2])
    def test_bezier_amp_allows_small_positive_k(self, signal, fn, k):
        out = fn(signal, k=k, variance=0.05, rng=np.random.default_rng(0))
        assert out.shape == signal.shape
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("fn", WARP_FNS)
    def test_non_integer_k_rejected(self, signal, fn):
        with pytest.raises(ValueError, match="k must be"):
            fn(signal, k=3.5, variance=0.01, rng=np.random.default_rng(0))


class TestFreqfiltShortSignal:
    """B3d: short signals get a package-level error, not a SciPy internal one."""

    def test_short_signal_raises(self):
        with pytest.raises(ValueError, match="too short"):
            freqfilt(np.zeros(20), 0.1, 0.8)

    def test_message_avoids_scipy_internals(self):
        with pytest.raises(ValueError) as exc:
            freqfilt(np.zeros(20), 0.1, 0.8)
        message = str(exc.value)
        assert "padlen" not in message
        assert "20" in message

    def test_lowpass_minimum_is_lower_than_bandpass(self):
        # an order-4 lowpass needs > 15 samples; the bandpass needs > 27
        lowpassed = freqfilt(np.random.default_rng(0).standard_normal(20), 0.0, 0.8)
        assert lowpassed.shape == (20,)
        with pytest.raises(ValueError, match="too short"):
            freqfilt(np.zeros(20), 0.1, 0.8)

    def test_full_band_bypasses_length_check(self):
        x = np.zeros(5)
        np.testing.assert_array_equal(freqfilt(x, 0, 1), x)

    def test_long_signal_filters(self):
        out = freqfilt(np.random.default_rng(0).standard_normal(200), 0.1, 0.8)
        assert out.shape == (200,)


class TestBurstCount:
    """B2: n_bursts=0 means no bursts, not full-length broadband noise."""

    def test_zero_bursts_adds_nothing(self, signal):
        noisy, burst = burstify(signal, snr_db=10, n_bursts=0, rng=np.random.default_rng(0))
        assert np.count_nonzero(burst) == 0
        np.testing.assert_array_equal(noisy, signal)

    def test_negative_bursts_rejected(self, signal):
        with pytest.raises(ValueError, match="n_bursts"):
            burstify(signal, snr_db=10, n_bursts=-1, rng=np.random.default_rng(0))

    def test_coverage_increases_with_burst_count(self, signal):
        _, b1 = burstify(signal, snr_db=10, n_bursts=1, burst_width=10,
                         rng=np.random.default_rng(3))
        _, b5 = burstify(signal, snr_db=10, n_bursts=5, burst_width=10,
                         rng=np.random.default_rng(3))
        assert 0 < np.count_nonzero(b1) < np.count_nonzero(b5) < len(signal)


class TestResamplePool:
    """B1: resample_pool accepts an external array, as the README documents."""

    def test_ndarray_pool(self, signal):
        pool = np.random.default_rng(1).standard_normal(1000)
        noisy, noise = noisify(signal, snr_db=15, dist="resample", resample_pool=pool,
                               rng=np.random.default_rng(0))
        assert noisy.shape == signal.shape
        assert noise.shape == signal.shape

    def test_list_pool(self, signal):
        pool = list(np.random.default_rng(1).standard_normal(100))
        noisy, _ = noisify(signal, snr_db=15, dist="resample", resample_pool=pool,
                           rng=np.random.default_rng(0))
        assert noisy.shape == signal.shape

    def test_self_pool_still_works(self, signal):
        noisy, _ = noisify(signal, snr_db=15, dist="resample", resample_pool="self",
                           rng=np.random.default_rng(0))
        assert noisy.shape == signal.shape

    def test_missing_pool_rejected(self, signal):
        with pytest.raises(ValueError, match="resample_pool"):
            noisify(signal, snr_db=15, dist="resample", resample_pool=None,
                    rng=np.random.default_rng(0))

    def test_samples_are_drawn_from_the_pool(self, signal):
        # a two-valued pool must yield exactly two distinct noise levels
        pool = np.array([-1.0, 1.0])
        _, noise = noisify(signal, snr_db=15, dist="resample", resample_pool=pool,
                           zero_mean=False, rng=np.random.default_rng(0))
        assert len(np.unique(np.round(noise, 9))) == 2
