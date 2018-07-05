package a2.knn;

import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.api.java.UDF2;

public class CalcVectorDistanceUDF implements UDF2<Vector, Vector, Double> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public Double call(Vector v1, Vector v2) throws Exception {
		return Math.sqrt(Vectors.sqdist(v1, v2));
	}

}
