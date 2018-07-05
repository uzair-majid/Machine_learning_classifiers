package a2.knn;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.TreeMap;

import org.apache.hive.com.esotericsoftware.minlog.Log;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.ForeachPartitionFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.RelationalGroupedDataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import a2.KNNDriverApp;
import scala.Function1;
import scala.Tuple2;
import scala.collection.mutable.HashMap;
import scala.runtime.BoxedUnit;

public class KNNModel extends Model<KNNModel> {

	private static int k;
	private Dataset<Row> trainingDS;
	private String master = "local[*]";
	private static final String PCA_FEAT_COL_NAME = "pca_features";
	private static final String TRAINING_FEAT_COL_NAME = "training_features";
	private static Logger log = Logger.getLogger(KNNModel.class);

	KNNModel() {
		super();
	}

	KNNModel(int kv, String masterv) {
		this.master = masterv;
		this.k = kv;
	}

	public static int getK() {
		return k;
	}

	public static void setK(int k) {
		KNNModel.k = k;
	}

	public String getMaster() {
		return master;
	}

	public void setMaster(String master) {
		this.master = master;
	}

	private static final long serialVersionUID = -868537485201967L;

	@Override
	public String uid() {
		return KNNModel.class.getName() + serialVersionUID;
	}

	/**
	 * Experimental cartesian startegy
	 * 
	 * @param testDS
	 * @return
	 */
	private Dataset<Row> calculateCartesianOptimized(Dataset<?> testDS) {
		SparkSession spark = SparkSession.builder().getOrCreate();
		SparkConf conf = spark.sparkContext().env().conf();
		int exec = conf.getInt("spark.executor.num", 8);
		int cores = conf.getInt("spark.executor.cores", 5);
		testDS = testDS.repartition(exec * cores * 20, testDS.col("test_id"));
		// testDS.cache();
		//testDS.take(1);
		Dataset<Row> ret = testDS.crossJoin(trainingDS);
		//ret.take(1);
		// testDS.unpersist();//release
		return ret;
	}

	@Override
	public Dataset<Row> transform(Dataset<?> testDS) {

		SparkSession spark = SparkSession.builder().getOrCreate();
		// register UDF & UDAF
		spark.udf().register("calcVectorDistance", new CalcVectorDistanceUDF(), DataTypes.DoubleType);
		spark.udf().register("findKNearestNeighboursLabelByFrequency", new FindKNearestNeighboursLabelByFrequency(k));
		calculateCartesianOptimized(testDS).createOrReplaceTempView("CartesianData");

		Dataset<Row> tmp = spark.sql(
				"select test_id,training_id,training_label, calcVectorDistance(test_features,training_features) as distance  from CartesianData");
		// tmp.take(1);//trigger execution
		// testDS.unpersist();
		// tmp.groupByKey(newFunction1{}, );
		tmp.createOrReplaceTempView("DistanceCalculatedData");

		Dataset<Row> result = spark.sql(
				"select test_id,findKNearestNeighboursLabelByFrequency(distance,training_label) as prediction  FROM DistanceCalculatedData GROUP BY test_id");
		result.take(1);
		return result;
	}

	@Override
	public StructType transformSchema(StructType schema) {
		// add prediciotn column
		schema.add(new StructField("prediction", DataTypes.DoubleType, true, Metadata.empty()));
		return schema;
	}

	@Override
	public KNNModel copy(ParamMap exrta) {
		defaultCopy(exrta);
		this.explainParams();
		;
		return this;
	}

	KNNModel validateAndTrain(Dataset<?> trainDS) {
		// check for required columns
		HashSet<String> cols = new HashSet<>();
		cols.addAll(Arrays.asList(trainDS.columns()));

		if (!(cols.contains("training_id") && cols.contains("training_label"))) {
			// required columns missing throw exception
			throw new IllegalArgumentException(
					"Required columns missing : supplied columns in training data set ::" + cols.toString());
		}

		if (cols.contains(PCA_FEAT_COL_NAME) && cols.contains(TRAINING_FEAT_COL_NAME)) {
			trainDS.drop(TRAINING_FEAT_COL_NAME);// replace features with pcafeat
			trainDS.withColumnRenamed(PCA_FEAT_COL_NAME, TRAINING_FEAT_COL_NAME);
		}

		// this.trainingDS = (Dataset<Row>) trainDS.limit(1000);
		this.trainingDS = (Dataset<Row>) trainDS;

		// this.trainingDS.take(1);

		return this;
	}

}