package a2;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import a2.knn.KNN;

public class KNNDriverApp extends AbstractDriver {

	// from configuration .. initializing defaults if not provided
	private static int dimention = 80;
	private static int KNearestNeighbours = 60;

	private static void initialize() throws IOException {
		Properties prop = new Properties();
		

		SparkSession spark = SparkSession.builder().appName("tmp").getOrCreate();

		String p2 = spark.conf().get("spark.local.dir");

		System.out.print("PATH :: " + p2 + "\"KNN_Driver.properties");

		prop.load(new FileInputStream(p2 + "/KNN_Driver.properties"));
		spark.stop();

		initialize(prop);
		// get the property value and print it out
		KNearestNeighbours = Integer.parseInt(prop.getProperty("knn.driver.k"));
		dimention = Integer.parseInt(prop.getProperty("knn.driver.d"));

	}

	public static void main(String[] args) throws Exception {

		// initialize system
		initialize();

		// read training & test dataset
		Dataset<Row> trainDS = readTrainingDataSet();
		Dataset<Row> testDS = readTestDataSet();

		// use PCA to reduce dimentions
		PCA pcaTrain = new PCA().setInputCol("training_features").setOutputCol("pcaFeatures").setK(dimention);
		PCA pcaTest = new PCA().setInputCol("test_features").setOutputCol("pcaFeatures").setK(dimention);

		// reduce dimentions dataset
		trainDS = pcaTrain.fit(trainDS).transform(trainDS);
		testDS = pcaTest.fit(testDS).transform(testDS);
		log.info("KNN ##  PCA transformation completed ");

		// Set KNN in pipeline
		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] { new KNN().setK(KNearestNeighbours) });

		// fit training data & learn on test data
		log.info("KNN ## fitting trainining data then learning labels for for test data.... ");
		Dataset<Row> result = pipeline.fit(trainDS).transform(testDS);

		log.info("KNN Execution completed");
		log.info("KNN ##  Instrumentation called to compile metrics & present them in csv file for this execution");
		// Instrumentation.computeMetrics(result,testDS);

		log.info("KNN Execution completed computing measure and write output");
		result.orderBy(result.col("test_id")).as("Test Data Row ID").orderBy("test_id").cache().show(false);
		Instrumentation.computeMetricsKNN(result, testDS, KNearestNeighbours, dimention, outputPath);
		// / sparkSession.stop();
	}

	@Override
	protected void finalize() throws Throwable {
		super.finalize();
		sparkSession.stop();
	} // TODO Auto-generated method stub
}
