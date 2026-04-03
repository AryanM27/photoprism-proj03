from locust import HttpUser, task, between


SEARCH_QUERIES = [
    "sunset at the beach",
    "dog playing in park",
    "city skyline at night",
    "mountains with snow",
    "people at a birthday party",
]


class SearchUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(3)
    def search(self):
        query = SEARCH_QUERIES[self.environment.runner.user_count % len(SEARCH_QUERIES)]
        self.client.post(
            "/search",
            json={"query": query, "top_k": 10},
        )

    @task(1)
    def health(self):
        self.client.get("/health")
