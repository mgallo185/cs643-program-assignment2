EC2 Cluster Setup for Distributed Spark ML Training
🧾 Instance Configuration
Instance Type: t2.micro (Free-tier eligible)

AMI: Ubuntu Server 24.04 LTS (HVM), 64-bit (x86)

Number of Instances:

4 EC2 instances for Spark cluster training (1 master, 3 workers)

1 EC2 instance (optional) for testing the trained model (can reuse master)

Storage: 8 GiB (default gp3 EBS volume)

Key Pair: spark-keypair.pem

Security Group:

Allow inbound traffic on ports:

22 (SSH) — Source: your IP

8080, 7077, 4040 — Source: your IP or cluster private subnet

Outbound: allow all (default)
