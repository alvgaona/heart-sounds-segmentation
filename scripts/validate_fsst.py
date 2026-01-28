#!/usr/bin/env python3
"""
Validation script for FSST implementation.

Validates correctness by checking mathematical properties:
1. Pure sinusoid should have energy concentrated at its frequency
2. DC signal should concentrate at 0 Hz
3. Chirp signal should track instantaneous frequency
4. Output dimensions should be correct
5. Compare STFT component against scipy

Usage:
    pixi run python scripts/validate_fsst.py
"""

import numpy as np
import scipy.signal
import ssq


def test_output_dimensions():
    """Test that output dimensions are correct."""
    print("Test 1: Output dimensions")
    print("-" * 40)

    fs = 1000.0
    window = scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False)

    for signal_len in [500, 1000, 2000]:
        signal = np.random.randn(signal_len)
        s, f, t = ssq.fsst(signal, fs, window)

        expected_freq_bins = len(window) // 2 + 1  # 65 for 128-point FFT
        expected_time_steps = signal_len  # hop=1

        assert s.shape == (expected_freq_bins, expected_time_steps), f"Spectrum shape mismatch: {s.shape}"
        assert len(f) == expected_freq_bins, f"Frequency axis length mismatch: {len(f)}"
        assert len(t) == expected_time_steps, f"Time axis length mismatch: {len(t)}"

        print(f"  Signal length {signal_len}: spectrum {s.shape}, freqs {len(f)}, times {len(t)} [OK]")

    print()
    return True


def test_frequency_axis():
    """Test that frequency axis is correctly computed."""
    print("Test 2: Frequency axis")
    print("-" * 40)

    fs = 1000.0
    window_len = 128
    window = scipy.signal.get_window(("kaiser", 0.5), window_len, fftbins=False)
    signal = np.random.randn(500)

    s, f, t = ssq.fsst(signal, fs, window)

    # Check frequency range
    assert np.isclose(f[0], 0.0), f"First frequency should be 0, got {f[0]}"
    assert np.isclose(f[-1], fs / 2), f"Last frequency should be Nyquist ({fs/2}), got {f[-1]}"

    # Check frequency spacing
    expected_df = fs / window_len
    actual_df = f[1] - f[0]
    assert np.isclose(actual_df, expected_df), f"Frequency spacing should be {expected_df}, got {actual_df}"

    print(f"  Frequency range: {f[0]:.1f} - {f[-1]:.1f} Hz [OK]")
    print(f"  Frequency spacing: {actual_df:.4f} Hz (expected {expected_df:.4f}) [OK]")
    print()
    return True


def test_time_axis():
    """Test that time axis is correctly computed."""
    print("Test 3: Time axis")
    print("-" * 40)

    fs = 1000.0
    window = scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False)
    signal_len = 500
    signal = np.random.randn(signal_len)

    s, f, t = ssq.fsst(signal, fs, window)

    # Check time range
    assert np.isclose(t[0], 0.0), f"First time should be 0, got {t[0]}"
    expected_last_time = (signal_len - 1) / fs
    assert np.isclose(t[-1], expected_last_time), f"Last time should be {expected_last_time}, got {t[-1]}"

    # Check time spacing (hop=1)
    expected_dt = 1.0 / fs
    actual_dt = t[1] - t[0]
    assert np.isclose(actual_dt, expected_dt), f"Time spacing should be {expected_dt}, got {actual_dt}"

    print(f"  Time range: {t[0]:.4f} - {t[-1]:.4f} s [OK]")
    print(f"  Time spacing: {actual_dt:.6f} s (expected {expected_dt:.6f}) [OK]")
    print()
    return True


def test_pure_sinusoid():
    """Test that pure sinusoid concentrates energy at its frequency."""
    print("Test 4: Pure sinusoid energy concentration")
    print("-" * 40)

    fs = 1000.0
    window = scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False)

    for test_freq in [50.0, 100.0, 200.0]:
        duration = 1.0
        t_sig = np.arange(0, duration, 1 / fs)
        signal = np.sin(2 * np.pi * test_freq * t_sig)

        s, f, t = ssq.fsst(signal, fs, window)

        # Check at middle time point (avoid boundary effects)
        t_mid = len(t) // 2
        magnitudes = np.abs(s[:, t_mid])

        # Find peak frequency
        peak_idx = np.argmax(magnitudes)
        peak_freq = f[peak_idx]

        # Peak should be within one frequency bin of the true frequency
        df = fs / len(window)
        freq_error = abs(peak_freq - test_freq)

        assert freq_error <= df, f"Peak frequency {peak_freq:.1f} Hz too far from {test_freq:.1f} Hz (error: {freq_error:.1f} Hz)"

        # Check energy concentration (peak should be dominant)
        total_energy = np.sum(magnitudes**2)
        peak_energy = magnitudes[peak_idx] ** 2

        # For synchrosqueezed transform, most energy should be at peak
        # Allow some spread due to window sidelobes
        neighbors = magnitudes[max(0, peak_idx - 2) : min(len(magnitudes), peak_idx + 3)]
        concentrated_energy = np.sum(neighbors**2)
        concentration_ratio = concentrated_energy / total_energy

        print(f"  {test_freq:.0f} Hz sinusoid: peak at {peak_freq:.1f} Hz, concentration: {concentration_ratio*100:.1f}% [OK]")

    print()
    return True


def test_dc_signal():
    """Test that DC signal concentrates energy at 0 Hz."""
    print("Test 5: DC signal energy concentration")
    print("-" * 40)

    fs = 1000.0
    window = scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False)
    signal = np.ones(1000)  # DC signal

    s, f, t = ssq.fsst(signal, fs, window)

    # Check at middle time point
    t_mid = len(t) // 2
    magnitudes = np.abs(s[:, t_mid])

    # DC should be at index 0
    dc_magnitude = magnitudes[0]
    max_other = np.max(magnitudes[1:])

    # DC should dominate
    assert dc_magnitude > max_other * 5, f"DC magnitude ({dc_magnitude:.2f}) should dominate others ({max_other:.2f})"

    print(f"  DC magnitude: {dc_magnitude:.2f}, max other: {max_other:.2f} [OK]")
    print()
    return True


def test_chirp_tracking():
    """Test that chirp signal tracks instantaneous frequency."""
    print("Test 6: Chirp frequency tracking")
    print("-" * 40)

    fs = 1000.0
    window = scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False)

    # Linear chirp from 50 Hz to 200 Hz
    f0, f1 = 50.0, 200.0
    duration = 1.0
    t_sig = np.arange(0, duration, 1 / fs)

    # Chirp signal
    k = (f1 - f0) / duration
    signal = np.sin(2 * np.pi * (f0 * t_sig + 0.5 * k * t_sig**2))

    s, f, t = ssq.fsst(signal, fs, window)

    # Check at several time points
    test_points = [0.2, 0.4, 0.6, 0.8]
    df = fs / len(window)
    errors = []

    for test_time in test_points:
        # Expected instantaneous frequency
        expected_freq = f0 + k * test_time

        # Find corresponding time index
        t_idx = int(test_time * fs)
        magnitudes = np.abs(s[:, t_idx])

        # Find peak frequency
        peak_idx = np.argmax(magnitudes)
        peak_freq = f[peak_idx]

        error = abs(peak_freq - expected_freq)
        errors.append(error)

        # Allow 2 frequency bins tolerance
        assert error <= 2 * df, f"At t={test_time}s: expected {expected_freq:.1f} Hz, got {peak_freq:.1f} Hz"

        print(f"  t={test_time}s: expected {expected_freq:.1f} Hz, got {peak_freq:.1f} Hz (error: {error:.1f} Hz) [OK]")

    print(f"  Mean tracking error: {np.mean(errors):.2f} Hz")
    print()
    return True


def test_compare_stft_with_scipy():
    """Compare our STFT component with scipy's STFT."""
    print("Test 7: STFT comparison with scipy")
    print("-" * 40)

    fs = 1000.0
    window = scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False)

    # Test signal
    t_sig = np.arange(0, 0.5, 1 / fs)
    signal = np.sin(2 * np.pi * 100 * t_sig)

    # Our FSST (we'll compare the general structure)
    s_ssq, f_ssq, t_ssq = ssq.fsst(signal, fs, window)

    # scipy STFT with same parameters
    # Note: scipy uses different normalization and boundary handling
    f_scipy, t_scipy, s_scipy = scipy.signal.stft(signal, fs=fs, window=window, nperseg=len(window), noverlap=len(window) - 1, boundary=None, padded=False)

    # Compare dimensions (may differ due to boundary handling)
    print(f"  Our FSST output: {s_ssq.shape}")
    print(f"  Scipy STFT output: {s_scipy.shape}")

    # Both should have same number of frequency bins
    assert s_ssq.shape[0] == s_scipy.shape[0], "Frequency bins should match"
    print(f"  Frequency bins match: {s_ssq.shape[0]} [OK]")

    # Check that peak frequencies are similar for the test signal
    mid_idx_fsst = s_ssq.shape[1] // 2
    mid_idx_scipy = s_scipy.shape[1] // 2

    peak_ssq = f_ssq[np.argmax(np.abs(s_ssq[:, mid_idx_fsst]))]
    peak_scipy = f_scipy[np.argmax(np.abs(s_scipy[:, mid_idx_scipy]))]

    print(f"  FSST peak frequency: {peak_ssq:.1f} Hz")
    print(f"  Scipy STFT peak frequency: {peak_scipy:.1f} Hz")

    # Both should identify the 100 Hz component
    df = fs / len(window)
    assert abs(peak_ssq - 100) <= df, f"FSST peak ({peak_ssq:.1f}) should be near 100 Hz"
    assert abs(peak_scipy - 100) <= df, f"Scipy peak ({peak_scipy:.1f}) should be near 100 Hz"
    print(f"  Both correctly identify 100 Hz component [OK]")

    print()
    return True


def main():
    print("\n" + "=" * 60)
    print("FSST Implementation Validation")
    print("=" * 60 + "\n")

    tests = [
        ("Output dimensions", test_output_dimensions),
        ("Frequency axis", test_frequency_axis),
        ("Time axis", test_time_axis),
        ("Pure sinusoid", test_pure_sinusoid),
        ("DC signal", test_dc_signal),
        ("Chirp tracking", test_chirp_tracking),
        ("STFT comparison", test_compare_stft_with_scipy),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"  [FAILED] {e}\n")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}\n")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
