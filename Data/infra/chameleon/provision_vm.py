# infra/chameleon/provision_vm.py
from chi import server, context, lease, network
import chi, os, datetime

context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")
username = os.getenv("USER")
project = "proj03"  # project naming convention on Chameleon

l = lease.Lease(f"lease-data-{project}-{username}", duration=datetime.timedelta(hours=8))
l.add_flavor_reservation(id=chi.server.get_flavor_id("m1.xlarge"), amount=1)
l.submit(idempotent=True)

s = server.Server(
    f"node-data-{project}-{username}",
    image_name="CC-Ubuntu24.04",
    flavor_name=l.get_reserved_flavors()[0].name,
)
s.submit(idempotent=True)
s.associate_floating_ip()
s.refresh()
s.show(type="widget")

security_groups = [
    {"name": "allow-ssh",   "port": 22,   "description": "SSH"},
    {"name": "allow-5432",  "port": 5432, "description": "Postgres"},
    {"name": "allow-5672",  "port": 5672, "description": "RabbitMQ AMQP"},
    {"name": "allow-15672", "port": 15672,"description": "RabbitMQ Management UI"},
    {"name": "allow-9000",  "port": 9000, "description": "MinIO API"},
    {"name": "allow-9001",  "port": 9001, "description": "MinIO Console"},
    {"name": "allow-5000",  "port": 5000, "description": "MLflow"},
    {"name": "allow-30234", "port": 30234,"description": "PhotoPrism NodePort"},
]
for sg in security_groups:
    chi.network.add_security_group_rules_if_not_present(sg)
    s.add_security_group(sg["name"])
print(f"VM ready. Floating IP: {s.floating_ip}")
