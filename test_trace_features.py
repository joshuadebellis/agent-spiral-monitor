import numpy as np
from scipy.spatial.distance import pdist, squareform

from trace_features import (
    ORIGINAL_PH_NAMES,
    baseline_features,
    cosine_distance_matrix,
    original_ph_features,
    persistence_diagrams,
    persistence_entropy,
    ph_features,
)


def euclidean_matrix(P):
    return squareform(pdist(P))


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


def test_circle_has_exactly_one_long_loop():
    t = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    P = np.c_[np.cos(t), np.sin(t)]
    dgms = persistence_diagrams(euclidean_matrix(P), maxdim=1)
    lifetimes = dgms[1][:, 1] - dgms[1][:, 0]
    assert int((lifetimes > 0.5).sum()) == 1
    assert ph_features(euclidean_matrix(P))["h1_max"] > 1.0


def test_line_has_no_loops():
    P = np.c_[np.linspace(0.0, 10.0, 20), np.zeros(20)]
    assert ph_features(euclidean_matrix(P))["h1_count"] == 0.0


def test_cosine_distance_matrix_basics():
    X = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    D = cosine_distance_matrix(X)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)
    assert np.isclose(D[0, 1], 0.0)
    assert np.isclose(D[0, 2], 1.0)


def test_entropy_of_single_bar_is_zero():
    assert persistence_entropy(np.array([0.7])) == 0.0


def test_single_point_trace_is_handled():
    X = np.array([[0.3, 0.4, 0.5]])
    assert all(v == 0.0 for v in ph_features(cosine_distance_matrix(X)).values())
    assert baseline_features(X, ["only step"])["n_points"] == 1.0
    assert original_ph_features(X)["count_h0"] == 1.0


def test_original_features_have_all_names_and_are_finite():
    t = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    f = original_ph_features(np.c_[np.cos(t), np.sin(t)])
    assert set(f) == set(ORIGINAL_PH_NAMES)
    assert all(np.isfinite(v) for v in f.values())


def test_two_far_clusters_give_two_components_at_unit_scale():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.01, size=(10, 3))
    b = rng.normal(0.0, 0.01, size=(10, 3)) + 10.0
    f = original_ph_features(np.vstack([a, b]))
    assert f["beta0_1.0"] == 2.0
    assert f["count_h0"] == 20.0


def test_sphere_has_a_void():
    assert original_ph_features(fibonacci_sphere(60))["count_h2"] >= 1.0



from trace_features import path_ph_features, recurrence_features


def test_path_complex_detects_return_to_start():
    t = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    X = np.c_[np.cos(t), np.sin(t)]
    f = path_ph_features(X)
    assert f["path_h1_count"] >= 1.0
    assert f["path_h1_max"] > 0.5


def test_path_complex_has_no_loop_on_one_way_arc():
    t = np.linspace(0.0, np.pi / 2.0, 24)
    X = np.c_[np.cos(t), np.sin(t)]
    assert path_ph_features(X)["path_h1_count"] == 0.0


def test_recurrence_counts_a_revisit():
    X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [1.0, 0.01, 0.0]])
    f = recurrence_features(X)
    assert f["revisit_frac"] > 0.0


def test_recurrence_is_zero_without_revisits():
    X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    assert recurrence_features(X)["revisit_frac"] == 0.0