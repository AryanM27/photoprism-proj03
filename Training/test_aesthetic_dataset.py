from src.common.config import load_config 
from src.datasets.uri_resolver import cache_manifest_from_uri 
from src.datasets.aesthetic_dataset import AestheticDataset

def main():     
    config = load_config("configs/aesthetic/linear_head_baseline.yaml")
    manifest_ref = config["dataset"].get("manifest_uri") or config["dataset"]["manifest_path"]
    manifest_path = cache_manifest_from_uri(config, manifest_ref)

    dataset = AestheticDataset(
         manifest_path=manifest_path,
         config=config,
         split="train",
         image_size=224,
     )

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print("Sample keys:", list(sample.keys()))
    print("Record ID:", sample["record_id"])
    print("Image ID:", sample["image_id"])
    print("Image ref:", sample["image_ref"])
    print("Resolved image path:", sample["resolved_image_path"])
    print("Aesthetic score:", sample["aesthetic_score"])
    print("Image tensor shape:", tuple(sample["image_tensor"].shape))

if __name__ == "__main__":
    main()

