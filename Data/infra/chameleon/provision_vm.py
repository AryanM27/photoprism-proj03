# infra/chameleon/provision_vm.py
# Targets KVM@TACC — virtualised Nova instances (m1.xlarge + floating IPs).
# Run AFTER provision_block.py so the Cinder volume exists to attach.
# Resource names use proj03 as suffix per course naming policy.
from chi import server, context, lease, network
import chi, os, datetime

context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")
username = os.getenv("USER")
project = "proj03"

print(f"User: {username} | Project: {project} | Site: KVM@TACC")

# --- Lease ---
l = lease.Lease(f"lease-{username}-{project}", duration=datetime.timedelta(hours=8))
l.add_flavor_reservation(id=chi.server.get_flavor_id("m1.xlarge"), amount=1)
l.submit(idempotent=True)
print("Lease submitted. Waiting for ACTIVE state...")
l.wait()
print(f"Lease '{l.name}' is ACTIVE.")

# --- Server ---
s = server.Server(
    f"node-{username}-{project}",
    image_name="CC-Ubuntu24.04",
    flavor_name=l.get_reserved_flavors()[0].name,
)
s.submit(idempotent=True)
s.associate_floating_ip()
s.refresh()
s.check_connectivity()
s.show(type="text")

# --- Security groups ---
security_groups = [
    {"name": "allow-ssh",   "port": 22,   "description": "Enable SSH traffic on TCP port 22"},
    {"name": "allow-5432",  "port": 5432, "description": "Enable TCP port 5432 (Postgres)"},
    {"name": "allow-5672",  "port": 5672, "description": "Enable TCP port 5672 (RabbitMQ AMQP)"},
    {"name": "allow-15672", "port": 15672,"description": "Enable TCP port 15672 (RabbitMQ Management UI)"},
    {"name": "allow-9000",  "port": 9000, "description": "Enable TCP port 9000 (MinIO API)"},
    {"name": "allow-9001",  "port": 9001, "description": "Enable TCP port 9001 (MinIO Console)"},
    {"name": "allow-5000",  "port": 5000, "description": "Enable TCP port 5000 (MLflow)"},
    {"name": "allow-2342",  "port": 2342, "description": "Enable TCP port 2342 (PhotoPrism)"},
    {"name": "allow-8080",  "port": 8080, "description": "Enable TCP port 8080 (Adminer)"},
    {"name": "allow-9090",  "port": 9090, "description": "Enable TCP port 9090 (Prometheus)"},
    {"name": "allow-3000",  "port": 3000, "description": "Enable TCP port 3000 (Grafana)"},
    {"name": "allow-6333",  "port": 6333, "description": "Enable TCP port 6333 (Qdrant REST API)"},
]

for sg in security_groups:
    secgroup = network.SecurityGroup(
        {
            "name": sg["name"],
            "description": sg["description"],
        }
    )
    secgroup.add_rule(direction="ingress", protocol="tcp", port=sg["port"])
    secgroup.submit(idempotent=True)
    s.add_security_group(sg["name"])

print(f"Updated security groups: {[sg['name'] for sg in security_groups]}")

# --- Attach block storage volume (created by provision_block.py) ---
cinder_client = chi.clients.cinder()
volumes = [v for v in cinder_client.volumes.list() if v.name == f"block-{username}-{project}"]
if not volumes:
    raise RuntimeError(
        f"Block volume 'block-{username}-{project}' not found. "
        "Run provision_block.py first."
    )
volume = volumes[0]
s.attach_volume(volume.id)
print(f"Block volume '{volume.name}' attached.")

# --- Clone repo + install Docker + Python via s.execute() ---
print("\nSetting up VM: cloning repo, installing Docker and Python...")
s.execute("git clone https://github.com/AryanM27/photoprism-proj03.git")
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
s.execute("sudo apt-get install -y python3-full python3-pip")
s.execute("cp photoprism-proj03/Data/docker/.env.example photoprism-proj03/Data/docker/.env")
print("VM setup complete.")

# --- Floating IP ---
s.refresh()
floating_ip = next(
    addr["addr"]
    for addrs in s.addresses.values()
    for addr in addrs
    if addr.get("OS-EXT-IPS:type") == "floating"
)
print(f"\nVM ready. Floating IP: {floating_ip}")
print(f"SSH: ssh cc@{floating_ip}")
print("\n--- First time only: format and mount the block volume ---")
print("  sudo parted -s /dev/vdb mklabel gpt && sudo parted -s /dev/vdb mkpart primary ext4 0% 100%")
print("  sudo mkfs.ext4 /dev/vdb1")
print("  sudo mkdir -p /mnt/block && sudo mount /dev/vdb1 /mnt/block")
print("  sudo chown -R cc /mnt/block && sudo chgrp -R cc /mnt/block")
print("  mkdir -p /mnt/block/postgres /mnt/block/rabbitmq /mnt/block/minio \\")
print("           /mnt/block/photoprism/storage /mnt/block/photoprism/originals")
print("\n--- Subsequent sessions: just mount ---")
print("  sudo mkdir -p /mnt/block && sudo mount /dev/vdb1 /mnt/block")
print("\n--- Fill in passwords then start services ---")
print("  vim ~/photoprism-proj03/Data/docker/.env")
print("  docker compose -f ~/photoprism-proj03/Data/docker/docker-compose.yml up -d")
