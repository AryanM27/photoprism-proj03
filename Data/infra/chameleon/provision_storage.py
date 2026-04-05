# infra/chameleon/provision_storage.py
# Targets CHI@TACC — Swift object storage for large data (raw images, DVC artifacts,
# MLflow model checkpoints). Not available on KVM@TACC.
# Resource names use proj03 as suffix per course naming policy.
"""
DEPRECATED: This script creates a Swift container (photoprism-proj03) that is no longer used.
Object storage has been migrated to Chameleon native S3 (CHI@TACC).
Use provision_storage.ipynb instead and follow the S3 setup cells.
# Replaced by Chameleon native S3 (CHI@TACC)
"""
import sys
print("ERROR: This script is deprecated. Use provision_storage.ipynb instead.")
print("Object storage is now accessed via Chameleon native S3 — see the notebook for setup.")
sys.exit(1)

from chi import context
import chi, swiftclient

context.version = "1.0"
context.choose_project()
context.choose_site(default="CHI@TACC")

print("Site: CHI@TACC")

os_conn = chi.clients.connection()
token = os_conn.authorize()
storage_url = os_conn.object_store.get_endpoint()

swift_conn = swiftclient.Connection(
    preauthurl=storage_url,
    preauthtoken=token,
    retries=5,
)

# Idempotent — put_container returns 202 if container already exists
container_name = "photoprism-proj03"
swift_conn.put_container(container_name)
print(f"Container '{container_name}' is ready.")
print("Browse at: https://chi.tacc.chameleoncloud.org/project/containers")
