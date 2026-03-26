from src.common.config import load_config
from src.datasets.manifests import load_and_validate_semantic_manifest


def main():
    config = load_config("Training/configs/semantic/openclip_vitb32_baseline.yaml")
    manifest_path = config["dataset"]["manifest_path"]

    records = load_and_validate_semantic_manifest(manifest_path)

    print(f"Loaded {len(records)} semantic records successfully.")
    print("First record:")
    print(records[0])


if __name__ == "__main__":
    main()