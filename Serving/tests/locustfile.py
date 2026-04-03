import io
import random
from locust import HttpUser, task, between
from PIL import Image


SEARCH_QUERIES = [
    "sunset at the beach",
    "dog playing in park",
    "city skyline at night",
    "mountains with snow",
    "people at a birthday party",
]

COLORS = [
    (135, 206, 235),
    (34, 139, 34),
    (255, 165, 0),
    (70, 130, 180),
    (220, 20, 60),
]


def make_test_image() -> bytes:
    color = random.choice(COLORS)
    img = Image.new("RGB", (224, 224), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class ServingUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(3)
    def search(self):
        query = random.choice(SEARCH_QUERIES)
        self.client.post(
            "/search",
            json={"query": query, "top_k": 10},
        )

    @task(2)
    def index(self):
        image_id = f"test_{random.randint(1000, 9999)}"
        image_bytes = make_test_image()
        self.client.post(
            "/index",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            data={"image_id": image_id},
        )

    @task(1)
    def health(self):
        self.client.get("/health")
