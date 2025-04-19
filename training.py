#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import re

def clean_column_names(df):
    """Clean column names by removing extra quotes and whitespace"""
    for old_col in df.columns:
        # Remove extra quotes and trim whitespace
        new_col = re.sub(r'["]+', '', old_col).strip()
        df = df.withColumnRenamed(old_col, new_col)
    return df

def main():
    # Initialize Spark Session for YARN cluster
    spark = SparkSession.builder \
        .appName("Wine Quality Prediction - Training") \
        .master("yarn") \
        .getOrCreate()
    
    # Parse command line arguments
    import sys
    if len(sys.argv) != 4:
        print("Usage: spark-submit wine_quality_trainer.py <trainingDataPath> <validationDataPath> <modelSavePath>")
        sys.exit(1)
    
    training_data_path = sys.argv[1]
    validation_data_path = sys.argv[2]
    model_save_path = sys.argv[3]
    
    try:
        # Load training data
        training_data = spark.read.csv(training_data_path, header=True, sep=';', inferSchema=True)
        validation_data = spark.read.csv(validation_data_path, header=True, sep=';', inferSchema=True)
        
        # Clean column names
        training_data = clean_column_names(training_data)
        validation_data = clean_column_names(validation_data)
        
        # Print schema to verify column names
        print("Training data schema:")
        training_data.printSchema()
        
        # Convert quality to integer type and rename to label
        training_data = training_data.withColumn("label", training_data["quality"].cast("integer"))
        validation_data = validation_data.withColumn("label", validation_data["quality"].cast("integer"))
        
        # Define features
        feature_cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
                      "chlorides", "free sulfur dioxide", "total sulfur dioxide", 
                      "density", "pH", "sulphates", "alcohol"]
        
        # Create vector assembler
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        
        # Create random forest classifier
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=20)
        
        # Create pipeline
        pipeline = Pipeline(stages=[assembler, rf])
        
        # Create parameter grid for cross validation
        paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [10, 20, 30]) \
            .addGrid(rf.maxDepth, [5, 10, 15]) \
            .build()
        
        # Create evaluator
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1")
        
        # Create cross validator
        crossval = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=3,
            parallelism=4)  # Using 4 parallel threads for 4 EC2 instances
        
        # Train model with cross validation
        print("Training model...")
        cv_model = crossval.fit(training_data)
        
        # Get best model
        best_model = cv_model.bestModel
        
        # Evaluate on validation data
        validation_predictions = best_model.transform(validation_data)
        f1_score = evaluator.evaluate(validation_predictions)
        print(f"Validation F1 Score: {f1_score}")
        
        # Print best model parameters
        rf_model = best_model.stages[-1]
        print(f"Best model parameters:")
        print(f"  Number of trees: {rf_model.getNumTrees}")
        print(f"  Max depth: {rf_model.getMaxDepth()}")
        
        # Save the model
        print(f"Saving model to {model_save_path}")
        best_model.save(model_save_path)
        
        print("Model training complete!")
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
    