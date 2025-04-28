package com.wine;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import java.io.IOException;

public class WineQualityTrainer {

    public static void main(String[] args) throws IOException {
        if (args.length < 3) {
            System.out.println("Usage: WineQualityTrainer <training-file> <validation-file> <model-output-path>");
            System.exit(1);
        }
        
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        String trainingFile = args[0];
        String validationFile = args[1];
        String modelPath = args[2];
        
        // Create Spark session
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Prediction Training")
                // Remove the hardcoded master - it will use the one from spark-submit
                .getOrCreate();
                
        try {
            // Define schema for the CSV files
            StructType schema = DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField("fixed_acidity", DataTypes.DoubleType, false),
                DataTypes.createStructField("volatile_acidity", DataTypes.DoubleType, false),
                DataTypes.createStructField("citric_acid", DataTypes.DoubleType, false),
                DataTypes.createStructField("residual_sugar", DataTypes.DoubleType, false),
                DataTypes.createStructField("chlorides", DataTypes.DoubleType, false),
                DataTypes.createStructField("free_sulfur_dioxide", DataTypes.DoubleType, false),
                DataTypes.createStructField("total_sulfur_dioxide", DataTypes.DoubleType, false),
                DataTypes.createStructField("density", DataTypes.DoubleType, false),
                DataTypes.createStructField("pH", DataTypes.DoubleType, false),
                DataTypes.createStructField("sulphates", DataTypes.DoubleType, false),
                DataTypes.createStructField("alcohol", DataTypes.DoubleType, false),
                DataTypes.createStructField("quality", DataTypes.DoubleType, false)
            });

            // Load training data
            Dataset<Row> trainingData = spark.read()
                    .option("header", "true")
                    .option("delimiter", ";")
                    .schema(schema)
                    .csv(trainingFile);
                    
            // Load validation data
            Dataset<Row> validationData = spark.read()
                    .option("header", "true")
                    .option("delimiter", ";")
                    .schema(schema)
                    .csv(validationFile);
                    
            // Convert quality to label (integer)
            trainingData = trainingData.withColumn("label", 
                trainingData.col("quality").cast(DataTypes.IntegerType));
            validationData = validationData.withColumn("label", 
                validationData.col("quality").cast(DataTypes.IntegerType));
                
            // Define feature columns
            String[] featureCols = {
                "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                "pH", "sulphates", "alcohol"
            };
            
            // Create vector assembler
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(featureCols)
                    .setOutputCol("features");
            
            // Create logistic regression model
            LogisticRegression lr = new LogisticRegression()
                    .setMaxIter(10)
                    .setRegParam(0.3)
                    .setElasticNetParam(0.8)
                    .setLabelCol("label")
                    .setFeaturesCol("features");
            
            // Create pipeline
            Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {assembler, lr});
            
            // Train model
            System.out.println("Training model...");
            PipelineModel model = pipeline.fit(trainingData);
            
            // Evaluate model on validation data
            Dataset<Row> predictions = model.transform(validationData);
            
            // Evaluate model
            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("label")
                    .setPredictionCol("prediction")
                    .setMetricName("f1");
            
            double f1 = evaluator.evaluate(predictions);
            System.out.println("F1 score: " + f1);
            
            // Save model
            model.write().overwrite().save(modelPath);
            System.out.println("Model saved to: " + modelPath);
            
        // Save F1 score using Spark
        Dataset<String> f1Dataset = spark.createDataset(
            java.util.Collections.singletonList(String.format("F1 score: %.6f", f1)),
            org.apache.spark.sql.Encoders.STRING()
        );
        String f1OutputPath = modelPath + "_f1";
        f1Dataset.write().mode("overwrite").text(f1OutputPath);
        System.out.println("F1 score saved to: " + f1OutputPath);


        } finally {
            spark.stop();
        }
    }
}
