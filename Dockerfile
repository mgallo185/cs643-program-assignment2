FROM openjdk:11-jdk

# Install wget
RUN apt-get update && apt-get install -y wget

# Set Spark version
ENV SPARK_VERSION=3.3.2
ENV HADOOP_VERSION=3

# Download and install Spark
RUN wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

WORKDIR /app
RUN mkdir -p /data

COPY target/wine-quality-1.0-SNAPSHOT-jar-with-dependencies.jar /app/

CMD ["spark-submit", \
     "--class", "com.wine.WineQualityPredictor", \
     "--master", "local[*]", \
     "/app/wine-quality-1.0-SNAPSHOT-jar-with-dependencies.jar", \
     "/data/wine-model", \
     "/data/TestDataset.csv"]
     
