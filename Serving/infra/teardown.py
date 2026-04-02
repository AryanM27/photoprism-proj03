# Serving/infra/teardown.py
# Tears down the serving VM and lease on KVM@TACC.
# Does NOT delete object storage or block volumes — data is preserved.
# Run from the Chameleon Jupyter environment.

from chi import server, context, lease
import chi, os

context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")
username = os.getenv("USER")
project = "proj03"

print(f"User: {username} | Project: {project} | Site: KVM@TACC")

# --- Delete server ---
try:
    s = server.get_server(f"node-serving-{username}-{project}")
    print(f"Deleting server '{s.name}'...")
    s.delete()
    print("Server deleted.")
except Exception as e:
    print(f"Server not found or already deleted: {e}")

# --- Release lease ---
try:
    l = lease.get_lease(f"lease-serving-{username}-{project}")
    print(f"Deleting lease '{l.name}'...")
    l.delete()
    print("Lease released.")
except Exception as e:
    print(f"Lease not found or already deleted: {e}")

print("\nTeardown complete. VM and lease released.")
print("Note: No data was deleted — re-run provision_vm.py to get a fresh VM.")
