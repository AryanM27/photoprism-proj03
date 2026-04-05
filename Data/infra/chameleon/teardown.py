# infra/chameleon/teardown.py
# Detaches block volume (PRESERVED), then deletes VM + lease on KVM@TACC.
# NEVER deletes object storage (Swift) or block storage volume — data is persistent.
from chi import server, context, lease
import chi, os, time

context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")
username = os.getenv("USER")
project = "proj03"

print(f"User: {username} | Project: {project} | Site: KVM@TACC")
os_conn = chi.clients.connection()
cinder_client = chi.clients.cinder()

# --- Detach block volume (keep the volume, just detach from the VM) ---
try:
    volumes = [v for v in cinder_client.volumes.list() if v.name == f"block-{username}-{project}"]
    if volumes:
        volume = volumes[0]
        if volume.status == "in-use":
            s = server.get_server(f"node-{username}-{project}")
            s.detach_volume(volume.id)
            print(f"Block volume '{volume.name}' detached (preserved).")
        else:
            print(f"Block volume '{volume.name}' already detached.")
    else:
        print("No block volume found — skipping detach.")
except Exception as e:
    print(f"Could not detach block volume: {e}")

# --- Delete the VM server (boot volume and data volume are preserved) ---
try:
    srv = os_conn.compute.find_server(f"node-{username}-{project}")
    if srv:
        os_conn.compute.delete_server(srv, ignore_missing=True)
        print(f"Server node-{username}-{project} deleted.")
        print("Server deleted. Boot volume and data volume are preserved.")
    else:
        print(f"Server node-{username}-{project} not found — already deleted.")
except Exception as e:
    print(f"Could not delete server: {e}")

# --- Release the lease ---
try:
    l = lease.get_lease(f"lease-{username}-{project}")
    lease.delete_lease(l.id)
    print(f"Lease lease-{username}-{project} deleted.")
except Exception as e:
    print(f"Lease not found or already deleted: {e}")

print("\nTeardown complete. Block volume and object storage intact.")
