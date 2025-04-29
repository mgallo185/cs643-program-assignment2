FROM openjdk:11-jdk-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*

# Install Spark
ENV SPARK_VERSION=3.3.2
ENV HADOOP_VERSION=3
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Set environment variables
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

# Create app directory
WORKDIR /app

# Copy JAR file
COPY target/wine-quality-1.0-SNAPSHOT-jar-with-dependencies.jar /app/

# Create necessary directories
RUN mkdir -p /data
RUN mkdir -p /data/spark-share

# Command to run the prediction job
CMD ["spark-submit", \
     "--class", "com.wine.WineQualityPredictor", \
     "--master", "local[*]", \
     "wine-quality-1.0-SNAPSHOT-jar-with-dependencies.jar", \
     "/data/wine-model", \
     "/data/TestDataset.csv"]
