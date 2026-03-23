# infra/chameleon/provision_vm.py
# Targets KVM@TACC because this site provides virtualised Nova instances
# (m1.xlarge flavors) with floating IPs, which are not available on bare-metal CHI sites.
from chi import server, context, lease, network
import chi, os, datetime

context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")
username = os.getenv("USER")
project = "proj03"  # project naming convention on Chameleon

print(f"User: {username} | Project: {project} | Site: KVM@TACC")

# Reserve an m1.xlarge VM for 8 hours
l = lease.Lease(f"lease-data-{project}-{username}", duration=datetime.timedelta(hours=8))
l.add_flavor_reservation(id=chi.server.get_flavor_id("m1.xlarge"), amount=1)
l.submit(idempotent=True)
print("Lease submitted. Waiting for ACTIVE state...")
l.wait()
print(f"Lease '{l.name}' is ACTIVE.")

# Launch the VM instance
s = server.Server(
    f"node-data-{project}-{username}",
    image_name="CC-Ubuntu24.04",
    flavor_name=l.get_reserved_flavors()[0].name,
)
s.submit(idempotent=True)
s.associate_floating_ip()
s.refresh()
s.show(type="text")

# Attach security groups for all data stack services
security_groups = [
    {"name": "allow-ssh",   "port": 22,   "description": "Enable SSH traffic on TCP port 22"},
    {"name": "allow-5432",  "port": 5432, "description": "Enable TCP port 5432 (Postgres)"},
    {"name": "allow-5672",  "port": 5672, "description": "Enable TCP port 5672 (RabbitMQ AMQP)"},
    {"name": "allow-15672", "port": 15672,"description": "Enable TCP port 15672 (RabbitMQ Management UI)"},
    {"name": "allow-9000",  "port": 9000, "description": "Enable TCP port 9000 (MinIO API)"},
    {"name": "allow-9001",  "port": 9001, "description": "Enable TCP port 9001 (MinIO Console)"},
    {"name": "allow-5000",  "port": 5000, "description": "Enable TCP port 5000 (MLflow)"},
    {"name": "allow-30234", "port": 30234,"description": "Enable TCP port 30234 (PhotoPrism NodePort)"},
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

# Extract floating IP from addresses (s.floating_ip attribute does not exist)
floating_ip = next(
    addr["addr"]
    for addrs in s.addresses.values()
    for addr in addrs
    if addr.get("OS-EXT-IPS:type") == "floating"
)
print(f"\nVM ready. Floating IP: {floating_ip}")
print(f"SSH: ssh cc@{floating_ip}")
