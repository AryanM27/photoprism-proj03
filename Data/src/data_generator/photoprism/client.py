import logging
from uuid import uuid4

import requests

logger = logging.getLogger(__name__)


class PhotoprismClient:
    def __init__(self, base_url: str, username: str, password: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._session_id: str | None = None
        self._user_uid: str | None = None
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    def login(self) -> None:
        url = f"{self._base_url}/api/v1/session"
        resp = self._session.post(
            url,
            json={"username": self._username, "password": self._password},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._session_id = data["id"]
        self._user_uid = data.get("user", {}).get("UID", "")
        self._session.headers.update({"X-Session-ID": self._session_id})
        logger.info("Logged in as %s (uid=%s)", self._username, self._user_uid)

    def logout(self) -> None:
        if self._session_id is None:
            return
        url = f"{self._base_url}/api/v1/session/{self._session_id}"
        resp = self._session.delete(url, timeout=30)
        resp.raise_for_status()
        self._session_id = None
        logger.info("Logged out")

    def search_photos(self, query: str, count: int = 20, offset: int = 0) -> list[dict]:
        url = f"{self._base_url}/api/v1/photos"
        params = {
            "q": query,
            "count": count,
            "offset": offset,
            "merged": "true",
            "public": "true",
            "quality": 1,
        }
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json() or []

    def upload_photo(self, image_bytes: bytes, filename: str) -> str | None:
        upload_token = uuid4().hex
        upload_url = f"{self._base_url}/api/v1/users/{self._user_uid}/upload/{upload_token}"
        files = {"files": (filename, image_bytes, "image/jpeg")}
        resp = self._session.post(upload_url, files=files, timeout=60)
        resp.raise_for_status()
        import_url = f"{self._base_url}/api/v1/users/{self._user_uid}/upload/{upload_token}"
        self._session.put(import_url, json={}, timeout=60)
        return upload_token

    def search_semantic(self, query: str, count: int = 20) -> list[dict]:
        url = f"{self._base_url}/api/v1/photos/semantic"
        params = {"q": query, "count": count}
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json() or []

    def click_photo_semantic(self, image_id: str, query: str, score: float) -> None:
        url = f"{self._base_url}/api/v1/semantic/click"
        resp = self._session.post(
            url,
            json={"image_id": image_id, "query": query, "score": score},
            timeout=30,
        )
        resp.raise_for_status()

    def like_photo(self, uid: str) -> None:
        url = f"{self._base_url}/api/v1/photos/{uid}/like"
        resp = self._session.post(url, timeout=30)
        resp.raise_for_status()

    def like_photo_semantic(self, image_id: str, query: str, score: float) -> None:
        url = f"{self._base_url}/api/v1/semantic/like"
        resp = self._session.post(
            url,
            json={"image_id": image_id, "query": query, "score": score},
            timeout=30,
        )
        resp.raise_for_status()

    def unlike_photo(self, uid: str) -> None:
        url = f"{self._base_url}/api/v1/photos/{uid}/like"
        resp = self._session.delete(url, timeout=30)
        resp.raise_for_status()

    def get_albums(self) -> list[dict]:
        url = f"{self._base_url}/api/v1/albums"
        params = {"count": 100, "offset": 0}
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json() or []

    def browse_album(self, album_uid: str, count: int = 20) -> list[dict]:
        url = f"{self._base_url}/api/v1/albums/{album_uid}/photos"
        params = {"count": count, "offset": 0}
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json() or []

    def get_photos(self, count: int = 20, offset: int = 0) -> list[dict]:
        url = f"{self._base_url}/api/v1/photos"
        params = {
            "count": count,
            "offset": offset,
            "merged": "true",
            "quality": 1,
        }
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json() or []

    @property
    def username(self) -> str:
        return self._username

    @property
    def is_logged_in(self) -> bool:
        return self._session_id is not None
