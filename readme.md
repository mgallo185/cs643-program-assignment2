
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
Setting up Github and the repo by Generating an SSH key and cloning the repo
```bash
ssh-keygen -t ed25519 -C "your_email@example.com" 
```
```bash
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/id_ed25519 
```

```bash
 cat ~/.ssh/id_ed25519.pub
```
Copy that public key and add it to GitHub:
Go to GitHub → Settings → SSH and GPG keys → New SSH key.
Paste it there.

``` git clone git@github.com:mgallo185/cs643-program-assignment2.git ```

## Setting up NFS File Sharing

### Setting up NFS Server on Master Node
```bash
# Install NFS server packages on the master node
sudo apt update
sudo apt install nfs-kernel-server -y

# Create a directory to share
sudo mkdir -p /data/spark-share
sudo chown -R ubuntu:ubuntu /data/spark-share

# Move datasets to the shared directory
cp ~/TrainingDataset.csv /data/spark-share/
cp ~/ValidationDataset.csv /data/spark-share/

# Make the directory accessible for all cluster nodes
sudo bash -c 'echo "/data/spark-share *(rw,sync,no_subtree_check,no_root_squash)" >> /etc/exports'

# Apply the exports
sudo exportfs -a

# Restart NFS server
sudo systemctl restart nfs-kernel-server

```

### Set up NFS Clients on Worker Nodes

```bash
# Install NFS client packages
sudo apt update
sudo apt install nfs-common -y

# Create mount point
sudo mkdir -p /data/spark-share

# Mount the shared directory from the master
# Replace MASTER_PRIVATE_IP with the private IP of your master node
sudo mount MASTER_PRIVATE_IP:/data/spark-share /data/spark-share

# Make the mount persist across reboots
sudo bash -c 'echo "MASTER_PRIVATE_IP:/data/spark-share /data/spark-share nfs rw,sync,hard,intr 0 0" >> /etc/fstab'

# Set correct permissions
sudo chown -R ubuntu:ubuntu /data/spark-share

```

### Verify NFS Setup

On the master node:
```bash
# Create a test file
touch /data/spark-share/nfs-test-file
```
On each worker node:

```bash
# Check if the file is visible
ls -la /data/spark-share/
```
You should see the test file on all nodes.

## Creating the Java Spark Project 

On your master node and in the directory where the Git repo that we cloned
``mkdir -p /src/main/java/com/wine``

Inside that directory is where the src/main/java/com/wine/WineQualityPredictor.java and src/main/java/com/wine/WineQualityTrainer.java

Outside of that directory you can make the pom.xml file

## Building the Project and Running the Train Model Code
This is still all done on your master node.

Build the proejct using 

`` mvn clean package  ``

This will create a JAR file with dependencies at target/wine-quality-1.0-SNAPSHOT-jar-with-dependencies.jar.

```bash
MASTER_IP=$(hostname -i)
sed -i "s/MASTER_IP/$MASTER_IP/g" src/main/java/com/wine/WineQualityTrainer.java
mvn clean package

#ensure data directory exists
mkdir -p /data/spark-share/
```

To run the traning job on the cluster
```bash
spark-submit --class com.wine.WineQualityTrainer \
  --master spark://$MASTER_IP:7077 \
  --deploy-mode client \
  --executor-memory 1g \
  --executor-cores 1 \
  --driver-memory 1g \
  --conf spark.executor.memoryOverhead=256m \
  --conf spark.network.timeout=600s \
  --conf spark.executor.heartbeatInterval=120s \
  target/wine-quality-1.0-SNAPSHOT-jar-with-dependencies.jar \
  /data/spark-share/TrainingDataset.csv \
  /data/spark-share/ValidationDataset.csv \
  /data/spark-share/wine-model
```

To run the prediction job locally using manual Spark submit:

```bash
sudo chmod 777 /data

spark-submit --class com.wine.WineQualityPredictor   --master local[*]   target/wine-quality-1.0-SNAPSHOT-jar-with-dependencies.jar   /data/spark-share/wine-model   /data/spark-share/TestDataset.csv
```

To run the Prediction with Docker:

# Create directories for data and model
```bash
mkdir -p $(pwd)/data
mkdir -p $(pwd)/model
```

# Copy test dataset
```bash
cp /path/to/your/TestDataset.csv $(pwd)/data/
```

# Copy your entire model directory (assuming it's trained already)
```bash
cp -r /data/spark-share/wine-model/* $(pwd)/model/
 ```
# Build the docker container
```bash
docker build -t wine-quality-predictor .
```
# This is the critical part - mount both directories
```bash docker run -v $(pwd)/data:/data -v $(pwd)/model:/data/wine-model wine-quality-predictor

 ```


# Project Directory

```
CS643-PROGRAM-ASSIGNMENT2/
├── data/        # Data directory for the predictions code
│   ├── prediction_results.txt
│   └── TestDataset.csv
├── model/        # Directory for ML Model artifact
│   ├── metadata/
│   └── stages/
├── src
|    └── /main/java/com/wine/
│         ├── WineQualityPredictor.java # Predictor Code
│         └── WineQualityTrainer.java # Trainer Code
├── .gitignore     # specfies files and directories that git ignores
├── target         # complied code but this is not on github
├── Dockerfile      #Dockerfile 
├── pa2.pdf         # Assignment pdf file
├── pom.xml         # Maven Project Configuration file
├── readme.md          # this file 
├── spark-keypair.pem  # key pair
├── TestDataset.csv    # TestDataset which is the same as Validation
├── TrainingDataset.csv  # Given Dataset
└── ValidationDataset.csv  # given dataset

```
