"""Unit tests for synthetic feedback generator."""
from src.data_pipeline.feedback.synthetic import generate_feedback_session


def test_feedback_session_returns_correct_count():
    image_ids = [f"img_{i}" for i in range(50)]
    events = generate_feedback_session("u_test", "sunset", image_ids, n_shown=10)
    assert len(events) == 10


def test_feedback_event_has_all_required_fields():
    image_ids = [f"img_{i}" for i in range(50)]
    events = generate_feedback_session("u1", "beach", image_ids)
    required = {
        "event_id", "user_id", "query_id", "image_id", "shown_rank",
        "clicked", "favorited", "semantic_score",
        "model_version", "timestamp",
    }
    assert all(required <= set(e.keys()) for e in events)


def test_all_events_share_same_query_id():
    image_ids = [f"img_{i}" for i in range(50)]
    events = generate_feedback_session("u1", "mountains", image_ids, n_shown=8)
    query_ids = {e["query_id"] for e in events}
    assert len(query_ids) == 1


def test_each_event_has_unique_event_id():
    image_ids = [f"img_{i}" for i in range(50)]
    events = generate_feedback_session("u1", "city", image_ids, n_shown=10)
    event_ids = [e["event_id"] for e in events]
    assert len(event_ids) == len(set(event_ids))


def test_shown_ranks_are_sequential():
    image_ids = [f"img_{i}" for i in range(50)]
    events = generate_feedback_session("u1", "food", image_ids, n_shown=10)
    assert [e["shown_rank"] for e in events] == list(range(10))


def test_feedback_top_results_have_higher_click_rate():
    image_ids = [f"img_{i}" for i in range(50)]
    top_clicks = bottom_clicks = 0
    for _ in range(200):
        events = generate_feedback_session("u", "q", image_ids, n_shown=10)
        top_clicks    += sum(1 for e in events[:3]  if e["clicked"])
        bottom_clicks += sum(1 for e in events[7:]  if e["clicked"])
    assert top_clicks > bottom_clicks


def test_n_shown_capped_at_pool_size():
    image_ids = [f"img_{i}" for i in range(5)]
    events = generate_feedback_session("u1", "test", image_ids, n_shown=20)
    assert len(events) == 5



def test_model_version_propagated():
    image_ids = [f"img_{i}" for i in range(50)]
    events = generate_feedback_session("u1", "q", image_ids, model_version="clip-v2")
    assert all(e["model_version"] == "clip-v2" for e in events)
