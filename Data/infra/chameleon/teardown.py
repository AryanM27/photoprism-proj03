# infra/chameleon/teardown.py
# Releases VM and lease only. NEVER deletes the object storage bucket.
from chi import server, context, lease
import chi, os

context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")
username = os.getenv("USER")
project = "proj03"

print(f"User: {username} | Project: {project} | Site: KVM@TACC")

# Delete the VM server
try:
    s = server.get_server(f"node-data-{project}-{username}")
    server.delete_server(s.id)
    print(f"Server node-data-{project}-{username} deleted.")
except Exception as e:
    print(f"Server not found or already deleted: {e}")

# Release the lease
try:
    l = lease.get_lease(f"lease-data-{project}-{username}")
    lease.delete_lease(l.id)
    print(f"Lease lease-data-{project}-{username} deleted.")
except Exception as e:
    print(f"Lease not found or already deleted: {e}")

print("\nTeardown complete. Object storage intact.")
