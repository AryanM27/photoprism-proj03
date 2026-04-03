from chi import server, context, lease
import chi, os

context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")
username = os.getenv("USER")
project = "proj03"

print(f"User: {username} | Project: {project} | Site: KVM@TACC")

try:
    s = server.get_server(f"node-serving-gpu-{username}-{project}")
    print(f"Deleting server '{s.name}'...")
    s.delete()
    print("Server deleted.")
except Exception as e:
    print(f"Server not found or already deleted: {e}")

try:
    l = lease.get_lease(f"lease-serving-gpu-{username}-{project}")
    print(f"Deleting lease '{l.name}'...")
    l.delete()
    print("Lease released.")
except Exception as e:
    print(f"Lease not found or already deleted: {e}")

print("\nGPU teardown complete.")
