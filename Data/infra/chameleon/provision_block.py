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

# ---------------------------------------------------------------------------
# Extend volume (optional — skip if no resize is needed)
# ---------------------------------------------------------------------------
import time as _time

NEW_SIZE_GIB = 40  # set to desired GiB — must be larger than current size

volume = cinder_client.volumes.get(volume.id)  # refresh to get latest size
if NEW_SIZE_GIB > volume.size:
    print(f"Extending volume from {volume.size} GiB to {NEW_SIZE_GIB} GiB...")
    os_conn = chi.clients.connection()
    os_conn.block_storage.extend_volume(volume.id, NEW_SIZE_GIB)
    while True:
        vol = os_conn.block_storage.get_volume(volume.id)
        if vol.status == "available":
            break
        elif vol.status == "error":
            raise RuntimeError("Volume extend failed — volume entered 'error' state.")
        _time.sleep(3)
    print(f"Volume extended to {NEW_SIZE_GIB} GiB.")
    # NOTE: if the volume is currently attached to a VM, you must also run
    #       `sudo resize2fs /dev/vdb1` inside the VM after extending.
else:
    print(f"Volume already at {volume.size} GiB — no resize needed.")
