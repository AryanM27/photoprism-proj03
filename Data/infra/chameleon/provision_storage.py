# infra/chameleon/provision_storage.py
# Targets CHI@TACC because Swift object storage (OpenStack) is only available
# at the bare-metal CHI sites, not on the KVM@TACC virtualised site.
import chi, os

chi.use_site("CHI@TACC")
chi.set("project_name", os.getenv("OS_PROJECT_NAME"))

container_name = "photoprism-data"
try:
    chi.swift.create_container(container_name)
    print(f"Object storage container created: {container_name}")
except Exception as e:
    if "409" in str(e) or "already exists" in str(e).lower():
        print(f"Container '{container_name}' already exists — skipping creation.")
    else:
        raise
print("Browse at: https://chi.tacc.chameleoncloud.org/project/containers")
