package com.wine;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.PCA;
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
import org.apache.spark.sql.functions;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.text.SimpleDateFormat;
import java.util.Date;

public class WineQualityTrainer {

    public static void main(String[] args) throws IOException {
        if (args.length < 3) {
            System.out.println("Usage: WineQualityTrainer <training-file> <validation-file> <model-output-path>");
            System.exit(1);
        }
        
        // Create timestamp for this run
        String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
        System.out.println("=================================================================");
        System.out.println("WINE QUALITY TRAINER RUN STARTED AT: " + timestamp);
        System.out.println("=================================================================");
        
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

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
            
            // Show some basic statistics of the data
            System.out.println("Training data summary:");
            trainingData.describe().show();
            
            // Convert quality to label (integer)
            trainingData = trainingData.withColumn("label", 
                trainingData.col("quality").cast(DataTypes.IntegerType));
            validationData = validationData.withColumn("label", 
                validationData.col("quality").cast(DataTypes.IntegerType));
            
            // Print class distribution
            System.out.println("Training Data Class Distribution:");
            trainingData.groupBy("label").count().orderBy("label").show();
            System.out.println("Validation Data Class Distribution:");
            validationData.groupBy("label").count().orderBy("label").show();
            
            // Feature Engineering: create additional features
            // 1. Ratio of free to total sulfur dioxide
            trainingData = trainingData.withColumn("free_to_total_so2_ratio", 
                functions.when(trainingData.col("total_sulfur_dioxide").equalTo(0.0), 0.0)
                .otherwise(trainingData.col("free_sulfur_dioxide").divide(trainingData.col("total_sulfur_dioxide"))));
                
            validationData = validationData.withColumn("free_to_total_so2_ratio", 
                functions.when(validationData.col("total_sulfur_dioxide").equalTo(0.0), 0.0)
                .otherwise(validationData.col("free_sulfur_dioxide").divide(validationData.col("total_sulfur_dioxide"))));
                
            // 2. Acidity to pH ratio
            trainingData = trainingData.withColumn("acidity_to_ph_ratio", 
                trainingData.col("fixed_acidity").divide(trainingData.col("pH")));
                
            validationData = validationData.withColumn("acidity_to_ph_ratio", 
                validationData.col("fixed_acidity").divide(validationData.col("pH")));
                
            // 3. Alcohol to density ratio
            trainingData = trainingData.withColumn("alcohol_to_density_ratio", 
                trainingData.col("alcohol").divide(trainingData.col("density")));
                
            validationData = validationData.withColumn("alcohol_to_density_ratio", 
                validationData.col("alcohol").divide(validationData.col("density")));
                
            // Original feature columns
            String[] originalFeatureCols = {
                "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                "pH", "sulphates", "alcohol"
            };
            
            // Engineered feature columns
            String[] engineeredFeatureCols = {
                "free_to_total_so2_ratio", "acidity_to_ph_ratio", "alcohol_to_density_ratio"
            };
            
            // Combine original and engineered features
            List<String> featureColsList = new ArrayList<>(Arrays.asList(originalFeatureCols));
            featureColsList.addAll(Arrays.asList(engineeredFeatureCols));
            String[] featureCols = featureColsList.toArray(new String[0]);
            
            // Create vector assembler
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(featureCols)
                    .setOutputCol("assembled_features");
            
            // Add feature scaling
            StandardScaler scaler = new StandardScaler()
                .setInputCol("assembled_features")
                .setOutputCol("scaled_features")
                .setWithStd(true)
                .setWithMean(true);
                
            // Optional: Add PCA for dimensionality reduction
            PCA pca = new PCA()
                .setInputCol("scaled_features")
                .setOutputCol("features")
                .setK(10);  // Number of principal components
                
            // Use RandomForestClassifier which supports multiclass classification
            RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setNumTrees(20)
                .setMaxDepth(10)
                .setSeed(42);
                
            // Create pipeline
            Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {assembler, scaler, pca, rf});
            
            // Create parameter grid for hyperparameter tuning
            // Fixed the parameter grid to work correctly with Java types
            ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(rf.maxDepth(), new int[] {5, 10, 15})
                .addGrid(rf.numTrees(), new int[] {10, 20, 30})
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
                .setNumFolds(5)  // 5-fold cross-validation
                .setSeed(42);
                
            // Train model with cross-validation
            System.out.println("Training model with cross-validation...");
            System.out.println("Cross-validation started at: " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
            CrossValidatorModel cvModel = cv.fit(trainingData);
            System.out.println("Cross-validation completed at: " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
            
            // Extract best model
            PipelineModel bestModel = (PipelineModel) cvModel.bestModel();
            
            // Evaluate model on validation data
            Dataset<Row> predictions = bestModel.transform(validationData);
            
            System.out.println("=================================================================");
            System.out.println("EVALUATION RESULTS FOR RUN STARTED AT: " + timestamp);
            System.out.println("=================================================================");
            
            // Calculate F1 score
            double f1 = evaluator.evaluate(predictions);
            System.out.println("F1 score on validation data: " + f1);
            
            // Calculate accuracy
            evaluator.setMetricName("accuracy");
            double accuracy = evaluator.evaluate(predictions);
            System.out.println("Accuracy on validation data: " + accuracy);
            
            // Calculate precision
            evaluator.setMetricName("weightedPrecision");
            double precision = evaluator.evaluate(predictions);
            System.out.println("Precision on validation data: " + precision);
            
            // Calculate recall
            evaluator.setMetricName("weightedRecall");
            double recall = evaluator.evaluate(predictions);
            System.out.println("Recall on validation data: " + recall);
            
            // Print confusion matrix
            System.out.println("Confusion Matrix:");
            predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show();
            
            // Save model
            bestModel.write().overwrite().save(modelPath);
            System.out.println("Model saved to: " + modelPath);
            
            // Save model evaluation metrics and configuration
            String metricsOutput = String.format(
                "Run Timestamp: %s\n" +
                "F1 Score: %.4f\n" +
                "Accuracy: %.4f\n" +
                "Precision: %.4f\n" +
                "Recall: %.4f\n" +
                "Best Parameters: %s",
                timestamp, f1, accuracy, precision, recall, 
                cvModel.bestModel().toString()
            );
            
            Dataset<String> metricsDataset = spark.createDataset(
                java.util.Collections.singletonList(metricsOutput),
                org.apache.spark.sql.Encoders.STRING()
            );
            
            String metricsPath = modelPath + "_metrics";
            metricsDataset.write().mode("overwrite").text(metricsPath);
            System.out.println("Metrics saved to: " + metricsPath);
            
            System.out.println("=================================================================");
            System.out.println("RUN COMPLETED AT: " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
            System.out.println("=================================================================");

        } finally {
            spark.stop();
        }
    }
}
