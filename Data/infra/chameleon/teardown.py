# infra/chameleon/teardown.py
# Releases VM and lease only. NEVER deletes the object storage bucket.
from chi import server, context, lease
import os

context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")
username = os.getenv("USER")
project = "proj03"

try:
    s = server.Server(f"node-data-{project}-{username}")
    s.delete()
    print(f"Server node-data-{project}-{username} deleted.")
except Exception as e:
    print(f"Server not found or already deleted: {e}")

try:
    l = lease.Lease(f"lease-data-{project}-{username}")
    l.delete()
    print(f"Lease lease-data-{project}-{username} deleted.")
except Exception as e:
    print(f"Lease not found or already deleted: {e}")

print("Teardown complete. Object storage intact.")
