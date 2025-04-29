
# Github Link: https://github.com/mgallo185/cs643-program-assignment2/
# Dockerhub Link: https://hub.docker.com/r/mgallo185/wine-quality-predictor

# Programming Assignment 2: Wine Quality Prediction Model

Goal: The purpose of this individual assignment is to learn how to develop parallel machine learning (ML) applications in Amazon AWS cloud platform. Specifically, you will learn: (1) how to use Apache Spark to train an ML model in parallel on multiple EC2 instances; (2) how to use Spark’s MLlib to develop and use an ML model in the cloud; (3) How to use Docker to create a container for your ML model to simplify model deployment.

Description: You have to build a wine quality prediction ML model in Spark over AWS. The model must be trained in parallel using 4 EC2 instances. Then, you need to save and load the model in a Spark application that will perform wine quality prediction; this application will run on one EC2 instance. The assignment must be implemented in Java on Ubuntu Linux. 

## Set up on AWS Console
1. Login and go to AWS Academy Learner Lab and click on Start Lab
2. Wait for AWS to load (when the small circle turns green) and click on AWS to access AWS Console
3. On AWS Console search and click on the EC2 service in the All Services Menu
4. On the EC2 Dashboard click Launch Instance
5. Configure Instance Details
  - Name: Give a name to your instance (Master-node for the master node and worker-number for each 3 worker nodes)
  - Ubuntu Linux: Ubunutu Server 24.04
  - Instance Type: t2.medium
  - Create New Key Pair give it a name and download the .pem file and save it in a safe place on your PC. (you will only need to do this once as you will use the same Key Pair for your other instance)
  - Configure Storage: 1x 16 GiB gp3
6. Configure Instance Security Group
   - Create a New Security Group and press the edit button
   - Give your security group a name
   - Allow the Necessary Ports to  Source Type **MY IP** and to Source Type **172.31.0.0/32**
   - Allow all traffic between instances in the same security group
   - Use the following Inbound Rules:
     | IP Version | Type | Protocol | Port Range | Source |
     |------------|------|----------|------------|--------|
     | - | All Traffic | All | All | Your-Security-Group |
     | IPv4 | SSH | TCP | 22 | Your-IP |
    
   - You will only need to do this once, when making your other instances, just use existing security group that you made


## Connecting to your EC2 Instances
1. Navigate in your terminal or Git Bash to where you downloaded the .pem file
2. Connect using SSH:
  - `chmod 400 my-key-pair.pem  # Set correct permissions`
  - `ssh -i my-key-pair.pem ec2-user@your-ec2-public-ip`
  - Replace **your-ec2-public-ip** with the Public IPv4 Address of your EC2 Dashboard
  - Do This to all your instances

## Setting Up the Enviroment on All Instances
SSH into each instance and run the following commands:
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Java (JDK 11)
sudo apt install openjdk-11-jdk -y

# Verify Java installation
java -version

# Install Maven for Java project management
sudo apt install maven -y

# Install Docker (on the  master node instance only)
sudo apt install docker.io -y
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu
```

## Setting Up Apache Spark
Run this on all instances

```bash
# Download and extract Spark
wget https://archive.apache.org/dist/spark/spark-3.3.2/spark-3.3.2-bin-hadoop3.tgz
tar -xvzf spark-3.3.2-bin-hadoop3.tgz
mv spark-3.3.2-bin-hadoop3 spark

# Set environment variables
echo "export SPARK_HOME=$HOME/spark" >> ~/.bashrc
echo "export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin" >> ~/.bashrc
echo "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64" >> ~/.bashrc
source ~/.bashrc

```

##  Configure Spark Cluster

### On the Master Node

```bash
# Start the master
$SPARK_HOME/sbin/start-master.sh

# Check the master UI is running
# Access http://<public-master-node-ip>:8080 in your browser
```
### On the 3 worker Nodes
```bash
# Start worker and connect to master
$SPARK_HOME/sbin/start-slave.sh spark://<private-master-node-ip>:7077
```

## Upload Datasets and Github Repo

Use SCP to upload the datasets to the master node
```bash
# From your local machine
scp -i your-key.pem TrainingDataset.csv ubuntu@<master-node-ip>:~/
scp -i your-key.pem ValidationDataset.csv ubuntu@<master-node-ip>:~/
```
Setting up Github and the repo by
1. Generate a new SSH key
```ssh-keygen -t ed25519 -C "your_email@example.com" ```
2.  ``` eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519 ```
3. ``` cat ~/.ssh/id_ed25519.pub ```
4. Copy that public key and add it to GitHub:

Go to GitHub → Settings → SSH and GPG keys → New SSH key.

Paste it there.

5. git clone git@github.com:mgallo185/cs643-program-assignment2.git




EC2 Cluster Setup for Distributed Spark ML Training
🧾 Instance Configuration
Instance Type: t2.medium

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

1. Terminate the t2.micro & Spin Up a t2.medium
Ubuntu 22.04 LTS AMI

Instance type: t2.medium or better

Open ports:

SSH (22) from your IP

Optional: Spark UI (4040)

☕ 2. Install Java & Set It Up
bash
Copy code
sudo apt update && sudo apt upgrade -y
sudo apt install openjdk-21-jdk git -y
Set environment variables:

bash
Copy code
echo 'export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))' >> ~/.bashrc
echo 'export PATH=$PATH:$JAVA_HOME/bin' >> ~/.bashrc
source ~/.bashrc
Confirm:

bash
Copy code
java -version
echo $JAVA_HOME
🔧 3. Install Apache Spark
bash
Copy code
wget https://dlcdn.apache.org/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
tar -xvzf spark-3.5.1-bin-hadoop3.tgz
sudo mv spark-3.5.1-bin-hadoop3 /opt/spark
Set Spark env:

bash
Copy code
echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc
echo 'export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin' >> ~/.bashrc
source ~/.bashrc
🛠️ 4. Set Up Your Java Project
Use Maven or Gradle.

Maven example:
bash
Copy code
mvn archetype:generate \
  -DgroupId=com.example.spark \
  -DartifactId=spark-java-project \
  -DarchetypeArtifactId=maven-archetype-quickstart \
  -DinteractiveMode=false
cd spark-java-project
Add Spark dependencies to pom.xml:
xml
Copy code
<dependencies>
  <dependency>
    <groupId>org.apache.spark</groupId>
    <artifactId>spark-core_2.12</artifactId>
    <version>3.5.1</version>
  </dependency>
  <dependency>
    <groupId>org.apache.spark</groupId>
    <artifactId>spark-sql_2.12</artifactId>
    <version>3.5.1</version>
  </dependency>
</dependencies>
Build it:

bash
Copy code
mvn clean package
🚀 5. Run Your Java Spark App
bash
Copy code
$SPARK_HOME/bin/spark-submit \
  --class com.example.spark.App \
  --master local[*] \
  target/spark-java-project-1.0-SNAPSHOT.jar
If you want, I can help you:

Write a sample Java Spark job to test everything

Set up IntelliJ or VS Code with remote dev

Switch to Gradle if you prefer that build system

What’s your next move — want help writing a Java Spark job from scratch, or you already have one you’re porting over?








