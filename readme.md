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








