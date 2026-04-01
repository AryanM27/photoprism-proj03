from src.common.config import load_config 
from src.datasets.uri_resolver import cache_manifest_from_uri  
config = load_config("configs/semantic/semantic_train_baseline.yaml") 
manifest_ref = config["dataset"].get("manifest_uri") or config["dataset"]["manifest_path"] 
manifest_path = cache_manifest_from_uri(config, manifest_ref)  
print("manifest_ref =", manifest_ref) 
print("manifest_path =", manifest_path)
