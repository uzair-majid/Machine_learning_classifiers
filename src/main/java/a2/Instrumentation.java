package a2;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.spark_project.guava.primitives.Doubles;

public class Instrumentation {

	static void computeMetricsKNN(Dataset<Row> result, Dataset<Row> testDS, int K, int d, String outputPath) { //
		result = result.select("test_id", "prediction");// keep double
		testDS = testDS.select("test_id", "test_label");
		StringBuffer buff = new StringBuffer();

		Dataset<Row> comparisionDS = result.join(testDS, "test_id");
		MulticlassMetrics metrics = new MulticlassMetrics(comparisionDS.select("test_label", "prediction"));
		buff.append(Integer.valueOf(K)).append(",").append(Integer.valueOf(d)).append(",--");

		writeMetrictoBuffer(metrics, buff);
		buff.append("--\r\n"); 
		List<String> list = new ArrayList<>();
		list.add("K,D,Execution Time,Accuracy,f_Score,Precision,Recall,other\r\n");
		list.add(buff.toString()); 
 
		for (String s : list) {
			System.out.println(s);
		}
		try {
			PrintWriter out = new PrintWriter(outputPath+"-metrics.csv");
			out.println("K,D,Execution Time,Accuracy,f_Score,Precision,Recall,other");
			out.println(buff.toString());
			System.out.println("K,D,Execution Time,Accuracy,f_Score,Precision,Recall,other");
			System.out.println(buff.toString());
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	 
	}

	static void writeMetrictoBuffer(MulticlassMetrics metrics, StringBuffer buff) {

		buff.append(",").append(metrics.accuracy());

		// f1 score
		buff.append(",-,");
		List<Double> list = Doubles.asList(metrics.labels());

		for (double label : list) {
			buff.append(label).append(":").append(metrics.fMeasure(label));buff.append(" | ");
		}
		buff.append(" , ");

		for (double label : list) {
			buff.append(label).append(":").append(metrics.precision(label));buff.append(" | ");
		}
		buff.append(" , ");

		for (double label : list) {
			buff.append(label).append(":").append(metrics.recall(label));buff.append(" | ");
		}
		 

	}

	static void computeMetricsNaiveBayes() {
		// TODO implement 
	}

}
