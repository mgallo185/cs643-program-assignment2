package com.wine;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
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
        // put loggers for errors
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);
        // Get training file, validation file and the model path
        String trainingFile = args[0];
        String validationFile = args[1];
        String modelPath = args[2];
        
        // Create Spark session
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Prediction Training")
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
                    
            // Convert quality to label 
            trainingData = trainingData.withColumn("label", 
                trainingData.col("quality").cast(DataTypes.IntegerType));
            validationData = validationData.withColumn("label", 
                validationData.col("quality").cast(DataTypes.IntegerType));
            
            // Print class distribution to understand data imbalance

            // organizes training data and validation data
            System.out.println("Training Data Class Distribution:");
            trainingData.groupBy("label").count().orderBy("label").show();
            System.out.println("Validation Data Class Distribution:");
            validationData.groupBy("label").count().orderBy("label").show();
            
            // Define all of the columns
            String[] featureCols = {
                "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                "pH", "sulphates", "alcohol"
            };
            
            // Create vector assembler
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(featureCols)
                    .setOutputCol("assembled_features");
            
            // Add feature scaling
            StandardScaler scaler = new StandardScaler()
                .setInputCol("assembled_features")
                .setOutputCol("features")
                .setWithStd(true)
                .setWithMean(true);
                
            // Try Random Forest classifier (better for imbalanced data)
            // Orgiinally used Logisitic but this gave a higher f1 value
            
            RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setNumTrees(100)
                .setMaxDepth(8)
                .setMaxBins(32)
                .setSeed(42)
                .setImpurity("gini");
            
        
            Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {assembler, scaler, rf});
            
            // Create parameter grid for hyperparameter tuning
            ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(rf.numTrees(), new int[] {50, 100})
                .addGrid(rf.maxDepth(), new int[] {5, 8, 10})
                .build();
                
            // Define evaluator
            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");
                
            // Create cross-validator
            CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(3)  // Use 3-fold cross-validation
                .setSeed(42);
                
            // Train model with cross-validation
            System.out.println("Training model with cross-validation...");
            CrossValidatorModel cvModel = cv.fit(trainingData);
            
            
            PipelineModel bestModel = (PipelineModel) cvModel.bestModel();
            
            // Evaluate model on validation data
            Dataset<Row> predictions = bestModel.transform(validationData);
            
            // Calculate F1 score
            double f1 = evaluator.evaluate(predictions);
            System.out.println("F1 score on validation data: " + f1);
            
            // Calculate accuracy
            evaluator.setMetricName("accuracy");
            double accuracy = evaluator.evaluate(predictions);
            System.out.println("Accuracy on validation data: " + accuracy);
            
            // Print confusion matrix
            System.out.println("Confusion Matrix:");
            predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show();
            
            // Save model to the path
            bestModel.write().overwrite().save(modelPath);
            System.out.println("Model saved to: " + modelPath);
            
            // Save model evaluation metrics
            String metricsOutput = String.format(
                "F1 Score: %.4f\nAccuracy: %.4f\nBest Parameters: %s",
                f1, accuracy, cvModel.bestModel().toString()
            );
            
            Dataset<String> metricsDataset = spark.createDataset(
                java.util.Collections.singletonList(metricsOutput),
                org.apache.spark.sql.Encoders.STRING()
            );
            //saves it in and overwrites a text file
            String metricsPath = modelPath + "_metrics";
            metricsDataset.write().mode("overwrite").text(metricsPath);
            System.out.println("Metrics saved to: " + metricsPath);

        } finally {
            spark.stop();
        }
    }
}
