package a2;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;

public class MPC {

	public static void main(String[] args) throws Exception {
		SparkSession spark = SparkSession.builder().master("yarn").appName("Java Spark Machine Learning multipreceptor")
				.getOrCreate();

		// String trainingSet = "input/Train-label-28x28.csv";
		// String testSet = "input/Test-label-28x28.csv";
		// String trainingSet = inputDataPath + "Train-label-28x28.csv";
		// String testSet = inputDataPath + "Test-label-28x28.csv";
		// if (args.length >= 2) {

		String trainingSet = args[0];
		String testSet = args[1];
		int dimension = Integer.parseInt(args[2]);
		int layervalue = Integer.parseInt(args[3]);
		int layervalue2 = Integer.parseInt(args[4]);

		Dataset<Row> df = spark.read().option("header", "false").option("inferSchema", true).csv(trainingSet); // training
		Dataset<Row> df2 = spark.read().option("header", "false").option("inferSchema", true).csv(testSet);// test

		df = df.withColumnRenamed("_c0", "label");

		df2 = df2.withColumnRenamed("_c0", "label");

		// scala.collection.immutable.List<StructField> Schemalist=
		// df.schema().toList();
		// Schemalist.drop(0);

		String temp[] = df.schema().fieldNames();
		List<String> templist = Arrays.asList(temp);

		String[] featureslist = templist.subList(1, templist.size()).toArray(new String[0]);

		// for(int i=0;i<featureslist.length;i++) {
		// //updatedschema[i]= Schemalist.take(i).toString();
		// System.out.println(featureslist[i]);
		// }

		// traininglabel= traininglabel.toDF("label2");

		// System.out.println("++++++++++++++++++++++++++"+df.count()+"+++++++++++++++++++++++++++++++++++++++");

		VectorAssembler assembler = new VectorAssembler().setInputCols(featureslist).setOutputCol("Features2");

		Dataset<Row> output = assembler.transform(df);
		Dataset<Row> output2 = assembler.transform(df2);

		// output.show(2);

		PCAModel model = new PCA().setK(dimension).setInputCol("Features2").setOutputCol("features").fit(output);
		Dataset<Row> trainingset = model.transform(output).select("label", "features");

		Dataset<Row> testset = model.transform(output2).select("label", "features");

		// specify layers for the neural network:
		// input layer of size 4 (features), two intermediate of size 5 and 4
		// and output of size 3 (classes)
		int[] layers = new int[] { dimension, layervalue, layervalue2, 20 };

		// create the trainer and set its parameters
		MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier().setLayers(layers)
				.setBlockSize(128).setSeed(1234L).setMaxIter(100);

		// train the model
		MultilayerPerceptronClassificationModel multimodel = trainer.fit(trainingset);

		// compute accuracy on the test set
		Dataset<Row> result = multimodel.transform(testset);
		Dataset<Row> predictionAndLabels = result.select("prediction", "label");
		// predictionAndLabels.show(10);
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy");

		predictionAndLabels.select("label", "prediction").show(5);
		MulticlassMetrics metrics = new MulticlassMetrics(
				predictionAndLabels.selectExpr("cast(label as double) label", "prediction"));

		org.apache.spark.mllib.linalg.Matrix confusion = metrics.confusionMatrix();

		//
		// Stats by labels
		for (int i = 0; i < metrics.labels().length; i++) {
			// System.out.format("Class %f precision = %f\n", metrics.labels()[i]));
			// System.out.format("Class %f recall = %f\n" );
			System.out.println("label: " + metrics.labels()[i] + " precision: " + metrics.precision(metrics.labels()[i])
					+ " recall: " + metrics.recall(metrics.labels()[i]) + " f1_Score "
					+ metrics.fMeasure(metrics.labels()[i]));
		}
		System.out.println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels));

		// spark.close();

	}
	// Load training data

}