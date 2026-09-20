import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import scipy.sparse as sp

from src.evaluation import beyond_accuracy as ba
from src.evaluation import ranking_metrics as rm


def test_precision_recall_f1_hit():
    rec, rel = [1, 2, 3, 4], {2, 4, 9}
    assert rm.precision_at_k(rec, rel, 4) == pytest.approx(0.5)
    assert rm.recall_at_k(rec, rel, 4) == pytest.approx(2 / 3)
    assert rm.f1_at_k(rec, rel, 4) == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert rm.hit_rate_at_k(rec, rel, 1) == 0.0
    assert rm.hit_rate_at_k(rec, rel, 2) == 1.0


def test_ndcg_perfect_and_hand_computed():
    assert rm.ndcg_at_k([1, 2, 3], {1, 2}, 3) == pytest.approx(1.0)
    # hit at rank 2 only, |rel|=1: DCG = 1/log2(3), IDCG = 1
    assert rm.ndcg_at_k([7, 1], {1}, 2) == pytest.approx(1 / np.log2(3))


def test_average_precision_hand_computed():
    # hits at ranks 1 and 3, |rel|=2: (1/1 + 2/3) / min(2, 3)
    assert rm.average_precision_at_k([1, 5, 2], {1, 2}, 3) == pytest.approx((1 + 2 / 3) / 2)


def test_k_truncates_the_list():
    assert rm.recall_at_k([5, 6, 1], {1}, 2) == 0.0


def test_no_rating_metrics_exposed():
    assert not any(name in rm.METRICS for name in ("rmse", "mae"))


def test_catalog_coverage_and_gini():
    assert ba.catalog_coverage([0, 1, 1, 2], 10) == pytest.approx(0.3)
    assert ba.gini_index([0, 1, 2, 3], 4) == pytest.approx(0.0)  # perfectly even exposure
    assert ba.gini_index([0, 0, 0, 0], 4) == pytest.approx(0.75)  # all exposure on one item


def test_artist_entropy_and_novelty_and_ild():
    artists = np.array(["a", "a", "b", "c"])
    assert ba.artist_entropy([0, 1], artists) == pytest.approx(0.0)
    assert ba.artist_entropy([0, 2], artists) == pytest.approx(1.0)
    assert ba.novelty([0, 1], np.array([0.5, 0.25])) == pytest.approx((1 + 2) / 2)
    content = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    assert ba.intra_list_diversity([0, 1], content) == pytest.approx(1.0)
    assert ba.intra_list_diversity([0, 2], content) == pytest.approx(0.0)
