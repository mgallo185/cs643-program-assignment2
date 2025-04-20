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

Fix Key Permissions (First Step After Downloading Key)
bash
Copy
Edit
chmod 400 spark-keypair.pem
🔌 SSH Into Instance
bash
Copy
Edit
ssh -i spark-keypair.pem ubuntu@<public-ip-address>
⚙️ EC2 Setup Script (Run on Each Instance)
bash
Copy
Edit
# Update & install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install openjdk-17-jdk python3-pip -y

# Download and install Spark
wget https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
tar -xvzf spark-3.5.1-bin-hadoop3.tgz
sudo mv spark-3.5.1-bin-hadoop3 /opt/spark

# Set environment variables (add to ~/.bashrc)
echo '
export SPARK_HOME=/opt/spark
export PATH=$SPARK_HOME/bin:$PATH
export JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:bin/java::")
' >> ~/.bashrc

# Reload shell
source ~/.bashrc
