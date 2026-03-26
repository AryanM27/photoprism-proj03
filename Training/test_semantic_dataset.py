from src.common.config import load_config
from src.datasets.semantic_dataset import SemanticRetrievalDataset


def main():
    config = load_config("Training/configs/semantic/openclip_vitb32_baseline.yaml")
    manifest_path = config["dataset"]["manifest_path"]

    dataset = SemanticRetrievalDataset(manifest_path=manifest_path, image_size=224)

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print("Sample keys:", list(sample.keys()))
    print("Image ID:", sample["image_id"])
    print("Text:", sample["text"])
    print("Image tensor shape:", tuple(sample["image_tensor"].shape))


if __name__ == "__main__":
    main()