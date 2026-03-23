# infra/chameleon/provision_block.py
# Creates a persistent Cinder block volume on KVM@TACC for small application state
# (PostgreSQL data, RabbitMQ, etc.). Run ONCE — volume survives VM teardowns.
# Resource names use proj03 as suffix per course naming policy.
from chi import context
import chi, os

context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")
username = os.getenv("USER")
project = "proj03"

print(f"User: {username} | Project: {project} | Site: KVM@TACC")

cinder_client = chi.clients.cinder()
volume_name = f"block-{username}-{project}"

# Idempotent — skip creation if volume already exists
existing = [v for v in cinder_client.volumes.list() if v.name == volume_name]
if existing:
    volume = existing[0]
    print(f"Volume '{volume_name}' already exists (status: {volume.status}) — skipping creation.")
else:
    volume = cinder_client.volumes.create(name=volume_name, size=20)
    print(f"Volume '{volume_name}' created (20 GiB). Waiting for AVAILABLE state...")
    import time
    while True:
        volume = cinder_client.volumes.get(volume.id)
        if volume.status == "available":
            break
        time.sleep(3)
    print(f"Volume '{volume_name}' is AVAILABLE.")

print(f"\nVolume ID: {volume.id}")
print("Run provision_vm.py next — it will attach this volume to the server.")
