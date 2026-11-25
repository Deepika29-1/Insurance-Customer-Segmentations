import pytest

from src import config
from src.generate_data import generate

pytestmark = pytest.mark.skipif(not (config.MODELS_DIR / "kmeans.joblib").exists(),
                                reason="Run `python run_pipeline.py` first to train the model")


def test_assign_segments_on_new_raw_customers():
    from src.predict import assign_segments, load_artifacts
    _, _, names = load_artifacts()
    out = assign_segments(generate(n=200, seed=99))
    assert out["segment"].notna().all()
    assert set(out["segment"]).issubset(set(names.values()))
