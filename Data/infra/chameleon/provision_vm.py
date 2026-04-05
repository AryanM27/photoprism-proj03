# infra/chameleon/provision_vm.py
# Targets KVM@TACC — virtualised Nova instances (m1.xlarge + floating IPs).
# Run AFTER provision_block.py so the Cinder volume exists to attach.
# Resource names use proj03 as suffix per course naming policy.
from chi import server, context, lease, network
import chi, os, datetime, time

context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")
username = os.getenv("USER")
project = "proj03"

print(f"User: {username} | Project: {project} | Site: KVM@TACC")
os_conn = chi.clients.connection()
cinder_client = chi.clients.cinder()

# --- Lease ---
l = lease.Lease(f"lease-{username}-{project}", duration=datetime.timedelta(hours=8))
l.add_flavor_reservation(id=chi.server.get_flavor_id("m1.xlarge"), amount=1)
l.submit(idempotent=True)
print("Lease submitted. Waiting for ACTIVE state...")
l.wait()
print(f"Lease '{l.name}' is ACTIVE.")

# ============================================================
# BOOT-FROM-VOLUME — persistent OS disk across lease renewals
# ============================================================
# Boot volume  = persistent OS disk (boot-vol-{username})
#   - Created once from CC-Ubuntu24.04 image (60 GiB)
#   - Survives lease expiry — same volume reused every cycle
#   - delete_on_termination is ALWAYS False
#
# Data volume  = existing block storage (block-{username}-proj03)
#   - Attached after boot as /dev/vdb, mounted at /mnt/block
#   - Attachment logic below is unchanged
#
# NEVER run volume.delete() on either unless intentionally
# wiping all project data and OS state.
# ============================================================
# Set to True only the very first time you set up this project.
# After the boot volume is created, set back to False.
RUN_ONE_TIME_SETUP = False

# ============================================================
# ONE-TIME SETUP — Run ONCE per project lifetime, then skip.
# Creates a 60 GiB Cinder volume from CC-Ubuntu24.04.
# After this cell completes, copy the printed volume ID and
# paste it into BOOT_VOL_ID in the EVERY-LEASE section below.
# ============================================================
if RUN_ONE_TIME_SETUP:
    image = os_conn.image.find_image("CC-Ubuntu24.04", ignore_missing=False)
    print(f"Found image: {image.name} ({image.id})")

    boot_vol = cinder_client.volumes.create(
        name=f"boot-vol-{username}",
        size=60,
        imageRef=image.id,
    )
    print(f"Creating boot volume '{boot_vol.name}' ({boot_vol.id}) — polling for availability...")

    while True:
        boot_vol = cinder_client.volumes.get(boot_vol.id)
        print(f"  boot volume status: {boot_vol.status}")
        if boot_vol.status == "available":
            break
        elif boot_vol.status == "error":
            raise RuntimeError(f"Boot volume entered error state: {boot_vol.id}")
        time.sleep(10)

    print(f"\nBoot volume ready.")
    print(f"Save this ID. Set BOOT_VOL_ID below: {boot_vol.id}")

# ============================================================
# EVERY-LEASE STARTUP — Fill in BOOT_VOL_ID once, then run
# this section every time you start a new lease.
# ============================================================
BOOT_VOL_ID = "FILL-IN-FROM-ONE-TIME-SETUP"   # <-- paste your volume ID here

if BOOT_VOL_ID == "FILL-IN-FROM-ONE-TIME-SETUP":
    raise ValueError("Set BOOT_VOL_ID to your boot volume ID before running this cell.")

server_name = f"node-{username}-{project}"
flavor_name = l.get_reserved_flavors()[0].name
flavor_id = chi.server.get_flavor_id(flavor_name)
network_id = os_conn.network.find_network("sharednet1").id

keypairs = list(os_conn.compute.keypairs())
if not keypairs:
    raise RuntimeError("No SSH keypairs found. Upload a keypair to KVM@TACC first.")
key_name = keypairs[0].name

bdm = [{
    "boot_index": 0,
    "uuid": BOOT_VOL_ID,
    "source_type": "volume",
    "destination_type": "volume",
    "delete_on_termination": False,
}]

srv = os_conn.compute.create_server(
    name=server_name,
    flavor_id=flavor_id,
    block_device_mapping_v2=bdm,
    networks=[{"uuid": network_id}],
    key_name=key_name,
)
print(f"Server '{server_name}' created ({srv.id}). Waiting for ACTIVE...")
srv = os_conn.compute.wait_for_server(srv)
print(f"Server '{server_name}' is ACTIVE.")

# Get chi wrapper for floating IP, security groups, and volume attachment
s = server.get_server(server_name)
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
# NOTE: These are first-boot only. Since the OS disk is persistent,
# subsequent lease cycles skip setup automatically if already installed.
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
print("           /mnt/block/photoprism/storage /mnt/block/photoprism/originals \\")
print("           /mnt/block/prometheus /mnt/block/grafana")
print("\n--- Subsequent sessions: just mount ---")
print("  sudo mkdir -p /mnt/block && sudo mount /dev/vdb1 /mnt/block")
print("\n--- Fill in passwords then start services ---")
print("  vim ~/photoprism-proj03/Data/docker/.env")
print("  docker compose -f ~/photoprism-proj03/Data/docker/docker-compose.yml up -d")
