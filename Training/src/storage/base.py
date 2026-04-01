# from abc import ABC, abstractmethod
# from typing import Any, Dict, List


# class StorageBackend(ABC):
#     @abstractmethod
#     def exists(self, path: str) -> bool:
#         pass

#     @abstractmethod
#     def makedirs(self, path: str) -> None:
#         pass

#     @abstractmethod
#     def save_text(self, path: str, content: str) -> None:
#         pass

#     @abstractmethod
#     def save_json(self, path: str, data: Dict[str, Any]) -> None:
#         pass

#     @abstractmethod
#     def load_json(self, path: str) -> Dict[str, Any]:
#         pass

#     @abstractmethod
#     def list_files(self, path: str) -> List[str]:
#         pass


from abc import ABC, abstractmethod
from typing import Any, Dict, List


class StorageBackend(ABC):
    @abstractmethod
    def exists(self, path: str) -> bool:
        pass

    @abstractmethod
    def makedirs(self, path: str) -> None:
        pass

    @abstractmethod
    def save_text(self, path: str, content: str) -> None:
        pass

    @abstractmethod
    def read_text(self, path: str) -> str:
        pass

    @abstractmethod
    def save_json(self, path: str, data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def load_json(self, path: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def list_files(self, path: str) -> List[str]:
        pass

    @abstractmethod
    def upload_file(self, local_path: str, remote_path: str) -> None:
        pass

    @abstractmethod
    def download_file(self, remote_path: str, local_path: str) -> None:
        pass