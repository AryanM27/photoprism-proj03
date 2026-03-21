# infra/chameleon/teardown.py
# Releases VM and lease only. NEVER deletes the object storage bucket.
from chi import server, context, lease
import os

context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")
username = os.getenv("USER")
project = "proj03"

s = server.Server(f"node-data-{project}-{username}")
s.delete()
l = lease.Lease(f"lease-data-{project}-{username}")
l.delete()
print("VM and lease released. Object storage intact.")
