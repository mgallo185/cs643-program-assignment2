package com.wine;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.functions;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.Date;

public class WineQualityPredictor {
    // Define a constant output file name
    // fixed generic output to a specfied directory data on where to put prediction results this output with a stored 
    private static final String OUTPUT_FILE = "/data/prediction_results.txt";    
    public static void main(String[] args) {
        // if arguments is less than 2 this is an error and exit the running of code you need 2 arguments
        if (args.length < 2) {
            System.out.println("Usage: WineQualityPredictor <model-path> <test-file>");
            System.exit(1);
        }
        
        // Set log levels to reduce noise make it easier to see if my code runs
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);
        // Get model Path as an argument
        String modelPath = args[0];
        // get file as an argument
        String testFile = args[1];
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(OUTPUT_FILE))) {
            // Log start of execution with current time 
            // time is get to check if the output updated because its hard to know if the output changed
            String currentTime = new Date().toString();
            logAndWrite(writer, "Starting Wine Quality Prediction at " + currentTime);
            logAndWrite(writer, "Model Path: " + modelPath);
            logAndWrite(writer, "Test File: " + testFile);
            
            // Create Spark session for prediction
            SparkSession spark = SparkSession.builder()
                    .appName("Wine Quality Prediction")
                    .getOrCreate();
                    
            try {
                // Define schema for the CSV files these are all rows from Validation/Test Dataset
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

                // Load test data
                logAndWrite(writer, "Loading test data from: " + testFile);
                Dataset<Row> testData = spark.read()
                        .option("header", "true")
                        .option("delimiter", ";")
                        .schema(schema)
                        .csv(testFile);
                
                logAndWrite(writer, "Number of test samples: " + testData.count());
                        
                // Convert quality to label (integer)
                testData = testData.withColumn("label", 
                    testData.col("quality").cast(DataTypes.IntegerType));
                
                // Look at test data distribution
                logAndWrite(writer, "Test Data Class Distribution:");
                String distributionStr = captureShowOutput(testData.groupBy("label").count().orderBy("label"), 10);
                logAndWrite(writer, distributionStr);
                    
                // Load model
                logAndWrite(writer, "Loading model from: " + modelPath);
                PipelineModel model = PipelineModel.load(modelPath);
                
                // Make predictions
                logAndWrite(writer, "Making predictions...");
                Dataset<Row> predictions = model.transform(testData);
                
                // Evaluate model using multiple metrics
                MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                        .setLabelCol("label")
                        .setPredictionCol("prediction");
                
                // Calculate F1 score (weighted)
                evaluator.setMetricName("f1");
                double f1 = evaluator.evaluate(predictions);
                logAndWrite(writer, "F1 score on test data: " + f1);
                
                // Calculate accuracy
                evaluator.setMetricName("accuracy");
                double accuracy = evaluator.evaluate(predictions);
                logAndWrite(writer, "Accuracy on test data: " + accuracy);
                
                // Calculate precision
                evaluator.setMetricName("weightedPrecision");
                double precision = evaluator.evaluate(predictions);
                logAndWrite(writer, "Precision on test data: " + precision);
                
                // Calculate recall
                evaluator.setMetricName("weightedRecall");
                double recall = evaluator.evaluate(predictions);
                logAndWrite(writer, "Recall on test data: " + recall);
                
                // Show sample predictions
                logAndWrite(writer, "Sample predictions:");
                String predictionSamples = captureShowOutput(predictions.select("quality", "prediction"), 15);
                logAndWrite(writer, predictionSamples);
                
                // Show confusion matrix
                logAndWrite(writer, "Confusion Matrix:");
                String confusionMatrix = captureShowOutput(
                    predictions.groupBy("label", "prediction").count().orderBy("label", "prediction"), 
                    50  // Show all possible combinations
                );
                logAndWrite(writer, confusionMatrix);
                
                // Count correct predictions and the accuracy use math to get the correct and divide by the total
                Dataset<Row> correctPredictions = predictions.filter(
                    predictions.col("prediction").equalTo(predictions.col("label")));
                long correctCount = correctPredictions.count();
                long totalCount = predictions.count();
                
                double accuracyPercent = (double)correctCount/totalCount * 100;
                logAndWrite(writer, String.format("Correct predictions: %d out of %d (%.2f%%)", 
                    correctCount, totalCount, accuracyPercent));
                
                // Check feature importance if available
                try {
                    logAndWrite(writer, "Feature Importance Analysis:");
                    // Try to extract feature importance (works if model contains RandomForestClassificationModel)
                    String modelString = model.toString();
                    if (modelString.contains("featureImportances")) {
                        String[] lines = modelString.split("\n");
                        for (String line : lines) {
                            if (line.contains("featureImportances")) {
                                logAndWrite(writer, line.trim());
                                break;
                            }
                        }
                    }
                } catch (Exception e) {
                    logAndWrite(writer, "Could not extract feature importance: " + e.getMessage());
                }
                
                logAndWrite(writer, "Prediction completed successfully.");
                
            } finally {
                spark.stop();
                logAndWrite(writer, "Spark session stopped.");
            }
            
            logAndWrite(writer, "Results saved to: " + OUTPUT_FILE);
            System.out.println("Results saved to: " + OUTPUT_FILE);
            
        } catch (IOException e) {
            System.err.println("Error writing to output file: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    // Log to both console and file
    private static void logAndWrite(PrintWriter writer, String message) {
        System.out.println(message);
        writer.println(message);
        writer.flush();
    }
    
    // Capture the output of DataFrame.show() method
    private static String captureShowOutput(Dataset<Row> df, int numRows) {
        StringBuilder sb = new StringBuilder();
        
        // Get column names
        String[] columns = df.columns();
        
        // Header row
        for (String column : columns) {
            sb.append(String.format("%-15s", column));
        }
        sb.append("\n");
        
        // Separator
        for (int i = 0; i < columns.length * 15; i++) {
            sb.append("-");
        }
        sb.append("\n");
        
        // Data rows
        Row[] rows = (Row[]) df.head(numRows);
        for (Row row : rows) {
            for (int i = 0; i < columns.length; i++) {
                sb.append(String.format("%-15s", row.get(i).toString()));
            }
            sb.append("\n");
        }
        
        return sb.toString();
    }
}
