import logging
import random
from pathlib import Path

import requests

from .augment import augment_image
from .client import PhotoprismClient
from .pipeline_bridge import register_upload
from .queries import random_query
from .s3_source import S3ImageSource
from .state import SimState

logger = logging.getLogger(__name__)

# Action weights — must sum to 1.0
# Skewed toward engagement (searches, clicks, likes) over uploads.
ACTION_WEIGHTS = {
    "browse": 0.10,
    "upload": 0.03,
    "favorite": 0.22,
    "semantic_search": 0.40,
    "click": 0.25,
}


def pick_action() -> str:
    return random.choices(
        list(ACTION_WEIGHTS.keys()),
        weights=list(ACTION_WEIGHTS.values()),
        k=1,
    )[0]


def do_search(client: PhotoprismClient, state: SimState) -> None:
    query = random_query()
    logger.debug("[%s] search: %s", state.user_id, query)
    photos = client.search_photos(query, count=random.randint(10, 30))
    uids = [p["UID"] for p in photos if "UID" in p]
    state.add_photos(uids)
    state.searches_done += 1


def do_semantic_search(client: PhotoprismClient, state: SimState) -> None:
    query = random_query()
    logger.debug("[%s] semantic_search: %s", state.user_id, query)
    try:
        results = client.search_semantic(query, count=random.randint(5, 20))
        for r in results:
            r["_query"] = query
        state.add_semantic_ids(results)
        state.searches_done += 1
    except requests.RequestException as exc:
        logger.warning("[%s] semantic search failed: %s", state.user_id, exc)


def _browse_album_safe(client: PhotoprismClient, state: SimState, album_uid: str, count: int) -> list[dict]:
    """Browse an album, evicting stale UIDs on 404."""
    try:
        photos = client.browse_album(album_uid, count=count)
        logger.debug("[%s] browse album %s -> %d photos", state.user_id, album_uid, len(photos))
        return photos
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            logger.debug("[%s] album %s not found, evicting", state.user_id, album_uid)
            if album_uid in state.seen_albums:
                state.seen_albums.remove(album_uid)
        else:
            raise
    return []


def do_browse(client: PhotoprismClient, state: SimState) -> None:
    # Prefer browsing a known album; fall back to recent photos feed
    photos: list[dict] = []
    if state.seen_albums:
        album_uid = random.choice(state.seen_albums)
        photos = _browse_album_safe(client, state, album_uid, count=random.randint(10, 30))

    if not photos and not state.seen_albums:
        # Refresh album list from server
        albums = client.get_albums()
        uids = [a["UID"] for a in albums if "UID" in a]
        state.add_albums(uids)
        if uids:
            album_uid = random.choice(uids)
            photos = _browse_album_safe(client, state, album_uid, count=random.randint(10, 30))

    if not photos:
        # No valid albums — browse recent photos
        photos = client.get_photos(count=random.randint(10, 30), offset=random.randint(0, 50))
        logger.debug("[%s] browse recent -> %d photos", state.user_id, len(photos))

    photo_uids = [p["UID"] for p in photos if "UID" in p]
    state.add_photos(photo_uids)
    state.browses_done += 1


def do_upload(
    client: PhotoprismClient, state: SimState, s3_source: S3ImageSource
) -> None:
    try:
        raw_bytes, orig_name = s3_source.random_image()
        aug_bytes, aug_name = augment_image(raw_bytes)
        aug_name = f"{Path(aug_name).stem}_datagen{Path(aug_name).suffix}"
        token = client.upload_photo(aug_bytes, aug_name)
        logger.debug(
            "[%s] uploaded %s (token=%s)", state.user_id, aug_name, token
        )
        image_id = register_upload(aug_bytes, aug_name, user_id=state.user_id)
        if image_id:
            state.uploads_done += 1
        else:
            logger.warning("[%s] upload indexed in Photoprism but not in ML pipeline", state.user_id)
    except (requests.RequestException, RuntimeError) as exc:
        logger.warning("[%s] upload failed: %s", state.user_id, exc)


def do_favorite(client: PhotoprismClient, state: SimState) -> None:
    result = state.random_semantic_result()
    if not result:
        logger.debug("[%s] favorite skipped — no semantic results seen", state.user_id)
        return
    try:
        client.like_photo_semantic(
            image_id=result.get("id", ""),
            query=result.get("_query", ""),
            score=result.get("score", 0.0),
        )
        logger.debug("[%s] liked semantic result %s", state.user_id, result.get("id"))
        state.likes_done += 1
    except requests.RequestException as exc:
        logger.warning("[%s] semantic like failed: %s", state.user_id, exc)


def do_click(client: PhotoprismClient, state: SimState) -> None:
    result = state.random_semantic_result()
    if not result:
        logger.debug("[%s] click skipped — no semantic results seen", state.user_id)
        return
    try:
        client.click_photo_semantic(
            image_id=result.get("id", ""),
            query=result.get("_query", ""),
            score=result.get("score", 0.0),
        )
        logger.debug("[%s] clicked %s", state.user_id, result.get("id"))
    except requests.RequestException as exc:
        logger.warning("[%s] click failed: %s", state.user_id, exc)


ACTION_FNS = {
    "browse": do_browse,
    "upload": do_upload,
    "favorite": do_favorite,
    "semantic_search": do_semantic_search,
    "click": do_click,
}
