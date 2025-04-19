#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import re

def clean_column_names(df):
    """Clean column names by removing extra quotes and whitespace"""
    for old_col in df.columns:
        # Remove extra quotes and trim whitespace
        new_col = re.sub(r'["]+', '', old_col).strip()
        df = df.withColumnRenamed(old_col, new_col)
    return df

def main():
    # Initialize Spark Session for single machine
    spark = SparkSession.builder \
        .appName("Wine Quality Prediction - Prediction") \
        .master("local[*]") \
        .getOrCreate()
    
    # Parse command line arguments
    import sys
    if len(sys.argv) != 3:
        print("Usage: spark-submit wine_quality_predictor.py <testDataPath> <modelPath>")
        sys.exit(1)
    
    test_data_path = sys.argv[1]
    model_path = sys.argv[2]
    
    try:
        # Load the model
        print(f"Loading model from {model_path}")
        model = PipelineModel.load(model_path)
        
        # Load test data
        print(f"Loading test data from {test_data_path}")
        test_data = spark.read.csv(test_data_path, header=True, sep=';', inferSchema=True)
        
        # Clean column names
        test_data = clean_column_names(test_data)
        
        # Print schema to verify column names
        print("Test data schema:")
        test_data.printSchema()
        
        # Convert quality to integer for evaluation
        test_data = test_data.withColumn("label", test_data["quality"].cast("integer"))
        
        # Make predictions
        print("Making predictions...")
        predictions = model.transform(test_data)
        
        # Show some example predictions
        print("Sample predictions:")
        predictions.select("label", "prediction", "features").show(10)
        
        # Evaluate model
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1")
        
        f1_score = evaluator.evaluate(predictions)
        print(f"F1 Score on test data: {f1_score}")
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()

if __name__ == "__main__":
    main()