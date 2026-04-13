"""
synthetic.py — Synthetic feedback session generator.

Simulates user query sessions: shows a ranked list of images, probabilistically
assigns clicks and favorites based on rank position (higher rank = more likely to click).

Usage:
    from src.data_pipeline.feedback.synthetic import generate_feedback_session
    events = generate_feedback_session("user_1", "sunset over water", image_ids)
"""

import uuid
import random
from datetime import datetime, timezone


def generate_feedback_session(
    user_id: str,
    query: str,
    image_ids: list[str],
    n_shown: int = 10,
    model_version: str = "v0",
) -> list[dict]:
    """Simulate one query session and return a list of feedback events.

    Click probability decays linearly with rank position:
        rank 0 → 40% click, rank 9 → ~4% click.
    Favorite probability is 20% of click probability.

    Args:
        user_id:       Synthetic user identifier.
        query:         Search query string.
        image_ids:     Pool of image IDs to sample from.
        n_shown:       Number of results shown in the session.
        model_version: Model version tag attached to each event.

    Returns:
        List of feedback event dicts, one per shown image.
    """
    shown = random.sample(image_ids, min(n_shown, len(image_ids)))
    query_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    events = []
    for rank, image_id in enumerate(shown):
        click_prob = max(0.05, 0.4 - rank * 0.04)
        fav_prob = click_prob * 0.2
        events.append({
            "event_id":        str(uuid.uuid4()),
            "user_id":         user_id,
            "query_id":        query_id,
            "image_id":        image_id,
            "shown_rank":      rank,
            "clicked":         random.random() < click_prob,
            "favorited":       random.random() < fav_prob,
            "semantic_score":  round(random.uniform(0.3, 1.0), 4),
            "model_version":   model_version,
            "timestamp":       now,
        })
    return events
