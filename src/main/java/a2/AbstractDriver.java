package a2;

import java.text.SimpleDateFormat;
import java.util.Properties;
import java.util.stream.Stream;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.SparkSessionExtensions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import io.netty.channel.local.LocalEventLoopGroup;
import scala.Tuple2;

public abstract class AbstractDriver {

	enum Master {
		YARN {
			public String toString() {
				return "yarn";
			}
		},

		LOCAL {
			public String toString() {
				return "local";
			}
		};
	}

	protected static Function<Tuple2<String, Long>, Row> parseFunction = new Function<Tuple2<String, Long>, Row>() {

		private static final long serialVersionUID = 1L;

		@Override
		public Row call(Tuple2<String, Long> t) throws Exception {
			String line = t._1;
			Long idx = t._2;
			int pos = line.indexOf(",");
			Object[] vals = new Object[3];
			vals[0] = idx;
			vals[1] = new Double(line.substring(0, pos));
			vals[2] = Vectors
					.dense(Stream.of(line.substring(pos + 1).split(",")).mapToDouble(Double::parseDouble).toArray())
					.compressed();
			return RowFactory.create(vals);
		}
	};

	protected static final String COMMA_DELIMITER_REGEX = ",(?=([^\"]*\"[^\"]*\")*[^\"]*$)";// regular expression to
	protected SparkSession spark;
	protected String applicationId;

	protected static final SimpleDateFormat dateFormatter = new SimpleDateFormat("yy.dd.mm");
	protected static Logger log = Logger.getLogger(AbstractDriver.class);

	// initializing default
	protected static String trainingInputFilePath  ;
	protected static String testInputFilePath ;
	protected static String outputPath;
	protected static SparkSession sparkSession = null;
	protected static String sparkExecutors = "1";
	protected static String execCore = "1";
	protected final static StructType schemaTrain = new StructType(
			new StructField[] { new StructField("training_id", DataTypes.LongType, false, Metadata.empty()),
					new StructField("training_label", DataTypes.DoubleType, false, Metadata.empty()),
					new StructField("training_features", new VectorUDT(), false, Metadata.empty()) });

	protected final static StructType schemaTest = new StructType(
			new StructField[] { new StructField("test_id", DataTypes.LongType, false, Metadata.empty()),
					new StructField("test_label", DataTypes.DoubleType, false, Metadata.empty()),
					new StructField("test_features", new VectorUDT(), false, Metadata.empty()) });

	protected static Master master = Master.LOCAL;

	protected static void initialize(Properties prop) {
		Logger.getLogger("org").setLevel(Level.ERROR);//make spark little quite 
		trainingInputFilePath = prop.getProperty("driver.trainingFile","input/Train-label-28x28-trim.csv");
		testInputFilePath = prop.getProperty("driver.testInputFile","input/Test-label-28x28-trim.csv");
		outputPath=prop.getProperty("driver.outputPrefix", "No-HDFS");
		// we need spark session to specify applicaion id for corect recording of data
		
		if(prop.getProperty("driver.master", Master.LOCAL.toString()).trim().toLowerCase().equals("yarn")){
			master=Master.YARN;
		}else {
			master=Master.LOCAL;
		}
		
		sparkExecutors = prop.getProperty("spark.executor.instances");
		execCore = prop.getProperty("spark.executor.cores");

		SparkConf conf = new SparkConf().setAppName("GROUP-117-8.Assignment2").setMaster(master.toString());
		conf.set("spark.shuffle.file.buffer", "3M");
		conf.set("spark.executor.memory", "2G");
		conf.set("spark.executor.instances", sparkExecutors);
		conf.set("spark.executor.cores", execCore); 
		sparkSession = SparkSession.builder().config(conf).getOrCreate();
		outputPath=new StringBuilder(outputPath).append(sparkSession.sparkContext().applicationId()).toString();
	} 
	protected static SparkSession getSparkSession() throws Exception {
		if (sparkSession == null)
			throw new Exception("Driver Not Initialized. Unable to obtain spark");
		return sparkSession;
	}

	protected static Dataset<Row> readTrainingDataSet() throws Exception {
		JavaSparkContext jsc = JavaSparkContext.fromSparkContext(getSparkSession().sparkContext());
		JavaRDD<Row> trainData = jsc.textFile(trainingInputFilePath, 3).filter(line -> line.indexOf(",") > 0)
				.zipWithIndex().map(parseFunction);
		return getSparkSession().createDataFrame(trainData, schemaTrain);
	}

	protected static Dataset<Row> readTestDataSet() throws Exception {
		JavaSparkContext jsc = JavaSparkContext.fromSparkContext(getSparkSession().sparkContext());
		JavaRDD<Row> testData = jsc.textFile(testInputFilePath, 3).filter(line -> line.indexOf(",") > 0).zipWithIndex()
				.map(parseFunction);
		return getSparkSession().createDataFrame(testData, schemaTest);
	}

}
