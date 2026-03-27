from src.common.config import load_config
from src.semantic.infer import generate_embeddings


def main():
    config = load_config("Training/configs/semantic/openclip_vitb32_baseline.yaml")

    outputs = generate_embeddings(
        manifest_path=config["dataset"]["manifest_path"],
        model_name=config["model"]["variant"],
        device_str=config["runtime"]["device"],
        use_mock_inference=config["model"].get("use_mock_inference", False),
        embedding_dim=config["model"].get("embedding_dim", 512),
    )

    print("Device:", outputs["device"])
    print("Image embeddings shape:", outputs["image_embeddings"].shape)
    print("Text embeddings shape:", outputs["text_embeddings"].shape)
    print("First image id:", outputs["image_ids"][0])
    print("First text:", outputs["texts"][0])


if __name__ == "__main__":
    main()