import random
import time
from dataclasses import dataclass, field


@dataclass
class SimState:
    user_id: str
    session_start: float = field(default_factory=time.time)

    # Recently seen photos (UIDs) from search/browse — used for like actions
    seen_photos: list[str] = field(default_factory=list)

    # Albums seen this session
    seen_albums: list[str] = field(default_factory=list)

    # Semantic IDs seen this session
    seen_semantic_ids: list[dict] = field(default_factory=list)

    # Counters
    searches_done: int = 0
    uploads_done: int = 0
    likes_done: int = 0
    browses_done: int = 0

    MAX_SEEN = 200  # cap to avoid unbounded growth

    def add_photos(self, uids: list[str]) -> None:
        self.seen_photos.extend(uids)
        if len(self.seen_photos) > self.MAX_SEEN:
            self.seen_photos = self.seen_photos[-self.MAX_SEEN:]

    def add_albums(self, uids: list[str]) -> None:
        self.seen_albums.extend(uids)
        if len(self.seen_albums) > 50:
            self.seen_albums = self.seen_albums[-50:]

    def add_semantic_ids(self, results: list[dict]) -> None:
        self.seen_semantic_ids.extend(results)
        if len(self.seen_semantic_ids) > self.MAX_SEEN:
            self.seen_semantic_ids = self.seen_semantic_ids[-self.MAX_SEEN:]

    def random_semantic_result(self) -> dict | None:
        if not self.seen_semantic_ids:
            return None
        return random.choice(self.seen_semantic_ids)

    def random_seen_photo(self) -> str | None:
        if not self.seen_photos:
            return None
        return random.choice(self.seen_photos)
