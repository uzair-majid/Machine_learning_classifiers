package a2.knn;

import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class KNN extends Estimator<KNNModel> {
	/**
	 * 
	 */
	private static final long serialVersionUID = 6426142641818115326L;
	Param<Integer> k = new Param<Integer>(this, "k", "set K nearest neighbours");
	Param<String> master = new Param<String>(this, "master", "set K nearest neighbours");

	public Param<Integer> getK() {
		return k;
	}

	public Param<String> getMaster() {
		return master;
	}

	public KNN() {
		super();
		setDefault(k, 10);
		setDefault(master, "local[*]");
	}

	@Override
	public String uid() {
		return getClass().getName() + serialVersionUID;
	}

	@Override
	public Estimator<KNNModel> copy(ParamMap extra) {
		return defaultCopy(extra);
	}

	public KNN setK(Integer k_) {
		set(k, k_);
		return this;
	}

	public KNN setMaster(String master_) {
		set(master, master_);
		return this;
	}

	@Override
	public KNNModel fit(Dataset<?> dataset) {
		Integer i = getOrDefault(k);
		String master_ = getOrDefault(master);
		return new KNNModel(i, master_).validateAndTrain(dataset);
	}

	@Override
	public StructType transformSchema(StructType schema) {
		schema.add(new StructField("prediction", DataTypes.DoubleType, true, Metadata.empty()));
		return schema;
	}

}
/*

*/
