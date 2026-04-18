import logging

import requests

from .client import PhotoprismClient
from .state import SimState

logger = logging.getLogger(__name__)


def bootstrap(client: PhotoprismClient) -> SimState:
    """Login and warm up initial state for this user session."""
    client.login()
    state = SimState(user_id=client.username)

    try:
        albums = client.get_albums()
        album_uids = [a["UID"] for a in albums if "UID" in a]
        state.add_albums(album_uids)
        logger.info(
            "Bootstrap: %d albums loaded for user %s",
            len(album_uids),
            client.username,
        )
    except requests.RequestException as exc:
        logger.warning("Bootstrap: failed to load albums: %s", exc)

    return state


def teardown(client: PhotoprismClient) -> None:
    try:
        client.logout()
    except Exception as exc:
        logger.warning("Teardown: logout failed: %s", exc)
