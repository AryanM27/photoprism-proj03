# infra/chameleon/provision_storage.py
import chi, os

chi.use_site("CHI@TACC")
chi.set("project_name", os.getenv("OS_PROJECT_NAME"))

container_name = "photoprism-data"
chi.storage.create_container(container_name)
print(f"Object storage container created: {container_name}")
print("Browse at: https://chi.tacc.chameleoncloud.org/project/containers")
