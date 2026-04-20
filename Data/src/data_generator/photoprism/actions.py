import logging
import random

import requests

from .augment import augment_image
from .client import PhotoprismClient
from .queries import random_query
from .s3_source import S3ImageSource
from .state import SimState

logger = logging.getLogger(__name__)

# Action weights — must sum to 1.0
ACTION_WEIGHTS = {
    "search": 0.35,
    "browse": 0.35,
    "upload": 0.15,
    "favorite": 0.15,
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
        token = client.upload_photo(aug_bytes, aug_name)
        logger.debug(
            "[%s] uploaded %s (token=%s)", state.user_id, aug_name, token
        )
        state.uploads_done += 1
    except (requests.RequestException, RuntimeError) as exc:
        logger.warning("[%s] upload failed: %s", state.user_id, exc)


def do_favorite(client: PhotoprismClient, state: SimState) -> None:
    uid = state.random_seen_photo()
    if not uid:
        logger.debug("[%s] favorite skipped — no seen photos", state.user_id)
        return
    try:
        client.like_photo(uid)
        logger.debug("[%s] liked %s", state.user_id, uid)
        state.likes_done += 1
    except requests.RequestException as exc:
        logger.warning("[%s] like %s failed: %s", state.user_id, uid, exc)


ACTION_FNS = {
    "search": do_search,
    "browse": do_browse,
    "upload": do_upload,
    "favorite": do_favorite,
}
