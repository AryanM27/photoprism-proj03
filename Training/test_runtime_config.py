from pprint import pprint

from src.common.config import load_config


def main():
    config = load_config("configs/semantic/openclip_vitb32_baseline.yaml")

    print("Resolved config:")
    pprint(config)

    print("\nResolved manifest path:")
    print(config["dataset"]["manifest_path"])

    print("\nResolved artifact dir:")
    print(config["output"]["artifact_dir"])

    if "checkpoint" in config:
        print("\nResolved checkpoint root:")
        print(config["checkpoint"]["root_dir"])

    print("\nRuntime section:")
    pprint(config.get("runtime", {}))

    print("\nStorage section:")
    pprint(config.get("storage", {}))


if __name__ == "__main__":
    main()