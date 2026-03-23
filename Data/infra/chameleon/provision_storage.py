# infra/chameleon/provision_storage.py
# Targets CHI@TACC — Swift object storage for large data (raw images, DVC artifacts,
# MLflow model checkpoints). Not available on KVM@TACC.
# Resource names use proj03 as suffix per course naming policy.
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
