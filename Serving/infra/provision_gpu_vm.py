from chi import server, context, lease, network
import chi, os, datetime

context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")
username = os.getenv("USER")
project = "proj03"

print(f"User: {username} | Project: {project} | Site: KVM@TACC")

l = lease.Lease(f"lease-serving-gpu-{username}-{project}", duration=datetime.timedelta(hours=8))
l.add_flavor_reservation(id=chi.server.get_flavor_id("g1.h100.pci.1"), amount=1)
l.submit(idempotent=True)
print("Lease submitted. Waiting for ACTIVE state...")
l.wait()
print(f"Lease '{l.name}' is ACTIVE.")

s = server.Server(
    f"node-serving-gpu-{username}-{project}",
    image_name="CC-Ubuntu24.04",
    flavor_name=l.get_reserved_flavors()[0].name,
)
s.submit(idempotent=True)
s.associate_floating_ip()
s.refresh()
s.check_connectivity()
s.show(type="text")

security_groups = [
    {"name": "allow-ssh",   "port": 22,   "description": "SSH"},
    {"name": "allow-8000",  "port": 8000, "description": "FastAPI serving endpoint"},
    {"name": "allow-6333",  "port": 6333, "description": "Qdrant vector DB"},
    {"name": "allow-9090",  "port": 9090, "description": "Prometheus metrics"},
    {"name": "allow-3000",  "port": 3000, "description": "Grafana dashboard"},
]

for sg in security_groups:
    secgroup = network.SecurityGroup({"name": sg["name"], "description": sg["description"]})
    secgroup.add_rule(direction="ingress", protocol="tcp", port=sg["port"])
    secgroup.submit(idempotent=True)
    s.add_security_group(sg["name"])

print(f"Security groups applied: {[sg['name'] for sg in security_groups]}")

print("\nSetting up VM: installing Docker and cloning repo...")
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
s.execute("git clone https://github.com/AryanM27/photoprism-proj03.git")
s.execute("sudo apt-get install -y nvidia-container-toolkit")
s.execute("sudo systemctl restart docker")
print("VM setup complete.")

s.refresh()
floating_ip = next(
    addr["addr"]
    for addrs in s.addresses.values()
    for addr in addrs
    if addr.get("OS-EXT-IPS:type") == "floating"
)
print(f"\nVM ready. Floating IP: {floating_ip}")
print(f"SSH: ssh cc@{floating_ip}")
print("\n--- Start serving stack ---")
print("  cd photoprism-proj03/Serving")
print("  docker compose up -d")
print("\n--- Endpoints ---")
print(f"  API:        http://{floating_ip}:8000")
print(f"  Prometheus: http://{floating_ip}:9090")
print(f"  Grafana:    http://{floating_ip}:3000")
