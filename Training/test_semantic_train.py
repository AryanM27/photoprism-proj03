from src.semantic.train import train_semantic_baseline


def main():
    summary = train_semantic_baseline(
        "Training/configs/semantic/semantic_train_baseline.yaml"
    )

    print("Semantic training summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()