from src.semantic.evaluate import run_semantic_evaluation


def main():
    config_path = "Training/configs/semantic/openclip_vitb32_baseline.yaml"
    metrics = run_semantic_evaluation(config_path)

    print("Semantic evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()