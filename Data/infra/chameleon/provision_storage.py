# infra/chameleon/provision_storage.py
# Targets CHI@TACC because Swift object storage (OpenStack) is only available
# at the bare-metal CHI sites, not on the KVM@TACC virtualised site.
from chi import context
import chi, swiftclient

context.version = "1.0"
context.choose_project()
context.choose_site(default="CHI@TACC")

print("Site: CHI@TACC")

# Connect to Swift using pre-authenticated credentials from the session
os_conn = chi.clients.connection()
token = os_conn.authorize()
storage_url = os_conn.object_store.get_endpoint()

swift_conn = swiftclient.Connection(
    preauthurl=storage_url,
    preauthtoken=token,
    retries=5,
)

# Create the object storage container (idempotent — put_container returns 202 if it already exists)
container_name = "photoprism-data"
swift_conn.put_container(container_name)
print(f"Container '{container_name}' is ready.")
print("Browse at: https://chi.tacc.chameleoncloud.org/project/containers")
