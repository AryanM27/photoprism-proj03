# from typing import Any, Dict, List

# from src.storage.base import StorageBackend


# class ObjectStoreBackend(StorageBackend):
#     """
#     Placeholder for future S3/MinIO-compatible implementation.
#     This keeps the interface stable while local backend remains the active implementation.
#     """

#     def __init__(self, endpoint: str = "", bucket: str = "", prefix: str = ""):
#         self.endpoint = endpoint
#         self.bucket = bucket
#         self.prefix = prefix

#     def exists(self, path: str) -> bool:
#         raise NotImplementedError("Object storage backend not implemented yet.")

#     def makedirs(self, path: str) -> None:
#         raise NotImplementedError("Object storage backend not implemented yet.")

#     def save_text(self, path: str, content: str) -> None:
#         raise NotImplementedError("Object storage backend not implemented yet.")

#     def save_json(self, path: str, data: Dict[str, Any]) -> None:
#         raise NotImplementedError("Object storage backend not implemented yet.")

#     def load_json(self, path: str) -> Dict[str, Any]:
#         raise NotImplementedError("Object storage backend not implemented yet.")

#     def list_files(self, path: str) -> List[str]:
#         raise NotImplementedError("Object storage backend not implemented yet.")


from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
import json

import swiftclient
import os

from src.storage.base import StorageBackend


def parse_swift_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "swift":
        raise ValueError(f"Unsupported object store URI: {uri}")

    container = parsed.netloc
    object_name = parsed.path.lstrip("/")

    if not container:
        raise ValueError(f"Missing Swift container in URI: {uri}")
    if not object_name:
        raise ValueError(f"Missing Swift object key in URI: {uri}")

    return container, object_name

def _build_swift_connection():
    auth_url = os.environ["OS_AUTH_URL"]
    auth_type = os.getenv("OS_AUTH_TYPE", "").strip()
    auth_version = os.getenv("OS_IDENTITY_API_VERSION", "3")

    os_options = {
        "region_name": os.getenv("OS_REGION_NAME") or os.getenv("OS_REGION"),
        "project_id": os.getenv("OS_PROJECT_ID"),
        "project_name": os.getenv("OS_PROJECT_NAME"),
        "user_domain_name": os.getenv("OS_USER_DOMAIN_NAME"),
        "project_domain_name": os.getenv("OS_PROJECT_DOMAIN_NAME"),
        "interface": os.getenv("OS_INTERFACE"),
        "auth_type": auth_type,
    }

    # remove empty values
    os_options = {k: v for k, v in os_options.items() if v}

    if auth_type == "v3applicationcredential":
        app_cred_id = os.getenv("OS_APPLICATION_CREDENTIAL_ID")
        app_cred_secret = os.getenv("OS_APPLICATION_CREDENTIAL_SECRET")

        if not app_cred_id or not app_cred_secret:
            raise ValueError(
                "OS_AUTH_TYPE is v3applicationcredential but application credential env vars are missing."
            )

        os_options["application_credential_id"] = app_cred_id
        os_options["application_credential_secret"] = app_cred_secret

        return swiftclient.Connection(
            authurl=auth_url,
            os_options=os_options,
            auth_version=auth_version,
        )

    # token auth path
    if auth_type == "token":
        auth_token = os.getenv("OS_AUTH_TOKEN") or os.getenv("OS_TOKEN")
        storage_url = os.getenv("OS_STORAGE_URL")

        if not auth_token:
            raise ValueError("Token auth selected but OS_AUTH_TOKEN / OS_TOKEN is missing.")
        if not storage_url:
            raise ValueError("Token auth selected but OS_STORAGE_URL is missing.")

        return swiftclient.Connection(
            preauthurl=storage_url,
            preauthtoken=auth_token,
            auth_version=auth_version,
            os_options=os_options,
        )

    # default username/password path
    username = os.getenv("OS_USERNAME")
    password = os.getenv("OS_PASSWORD")

    if not username or not password:
        raise ValueError(
            "Username/password auth selected but OS_USERNAME / OS_PASSWORD is missing."
        )

    return swiftclient.Connection(
        authurl=auth_url,
        user=username,
        key=password,
        os_options=os_options,
        auth_version=auth_version,
    )


class ObjectStoreBackend(StorageBackend):
    def __init__(
        self,
    ):
        self.conn = _build_swift_connection()

    def exists(self, path: str) -> bool:
        container, obj = parse_swift_uri(path)
        try:
            self.conn.head_object(container, obj)
            return True
        except swiftclient.exceptions.ClientException as e:
            if e.http_status == 404:
                return False
            raise

    def makedirs(self, path: str) -> None:
        # No-op for object storage
        return

    def save_text(self, path: str, content: str) -> None:
        container, obj = parse_swift_uri(path)
        self.conn.put_object(container, obj, contents=content.encode("utf-8"))

    def read_text(self, path: str) -> str:
        container, obj = parse_swift_uri(path)
        _, body = self.conn.get_object(container, obj)
        return body.decode("utf-8")

    def save_json(self, path: str, data: Dict[str, Any]) -> None:
        payload = json.dumps(data, indent=2).encode("utf-8")
        container, obj = parse_swift_uri(path)
        self.conn.put_object(container, obj, contents=payload)

    def load_json(self, path: str) -> Dict[str, Any]:
        return json.loads(self.read_text(path))

    def list_files(self, path: str) -> List[str]:
        container, prefix = parse_swift_uri(path)
        _, objects = self.conn.get_container(container, prefix=prefix)
        return [f"swift://{container}/{item['name']}" for item in objects]

    def upload_file(self, local_path: str, remote_path: str) -> None:
        container, obj = parse_swift_uri(remote_path)
        with open(local_path, "rb") as f:
            self.conn.put_object(container, obj, contents=f)

    def download_file(self, remote_path: str, local_path: str) -> None:
        container, obj = parse_swift_uri(remote_path)
        _, body = self.conn.get_object(container, obj)

        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            f.write(body)