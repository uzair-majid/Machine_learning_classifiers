package a2.knn;

import java.util.ArrayList;
import java.util.EmptyStackException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Stack;
import java.util.TreeMap;

import org.apache.hive.com.esotericsoftware.minlog.Log;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.expressions.MutableAggregationBuffer;
import org.apache.spark.sql.expressions.UserDefinedAggregateFunction;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class FindKNearestNeighboursLabelByFrequency extends UserDefinedAggregateFunction {

	private static final long serialVersionUID = -5355885686437887949L;
	private StructType inputSchema;
	private StructType bufferSchema;

	private final boolean deterministic = true;
	//
	// to hold label counts
	private final DataType dataType = DataTypes.DoubleType;

	private static Logger log = Logger.getLogger(FindKNearestNeighboursLabelByFrequency.class);

	private static Stack<TreeMap<Double, Double>> pool = new Stack<TreeMap<Double, Double>>();

	private final int k;

	public FindKNearestNeighboursLabelByFrequency(int k) {
		super();
		this.k = k;
		List<StructField> inputFields = new ArrayList<>();// we taking labels doubles
		inputFields.add(DataTypes.createStructField("distance", dataType, true));
		inputFields.add(DataTypes.createStructField("training_label", dataType, true));
		inputSchema = DataTypes.createStructType(inputFields);

		List<StructField> bufferFields = new ArrayList<>();
		bufferFields.add(DataTypes.createStructField("data", DataTypes.createMapType(dataType, dataType), false));
		bufferSchema = DataTypes.createStructType(bufferFields);

	}

	@Override
	public StructType bufferSchema() {
		return bufferSchema;
	}

	/**
	 * This is the data type of the output
	 */
	@Override
	public DataType dataType() {
		return dataType;
	}

	/**
	 * exact same out put for same input
	 */
	@Override
	public boolean deterministic() {
		return deterministic;
	}

	@Override
	public void initialize(MutableAggregationBuffer buffer) {
		buffer.update(0, new TreeMap<Double, Row>());
	}

	@Override
	public StructType inputSchema() {
		return inputSchema;
	}

	@Override
	public void update(MutableAggregationBuffer buffer, Row input) {
		// TODO: Include code snippet in report to show effiicient top K neighbour
		// Small footprint
		TreeMap<Double, Double> internalBuffer = new TreeMap<>();
		// update in internal buffer
		internalBuffer.putAll((Map<Double, Double>) buffer.<Double, Double>getJavaMap(0));

		Double distance = null;
		Double label = null;

		try {
			distance = input.getDouble(0);
			label = input.getDouble(1);

			if (distance == null || label == null) {// best practices , attemting to catch un expected problems
				throw new NullPointerException();
			}

		} catch (Exception e) {
			Object obj = input.get(0);
			log.error(
					"Unable to determine value for distance or label.. skip corrupt input.. if this line shows up in log investigate specially if more than once");
			Log.warn("Unexpected situation:: update method in FindKNearestNeighboursLabelByFreq received input object: "
					+ obj.getClass().toString());

			return;
		}

		// we look good at this point
		// add label & distance to internal buffer
		internalBuffer.put(distance, label);

		// trim buffer to keep top N values
		int size = internalBuffer.size();
		if (size > k) {
			// find & keep N only
			int rem = size - k;
			for (int x = size; x > k; x--) {
				internalBuffer.remove(internalBuffer.lastKey());// trim the tree to keep top K or smallest K values
			}
		}

		// we have top K values at this point for so far scanned input
		log.debug("Buffer from update " + buffer);
		buffer.update(0, internalBuffer);
	}

	@Override
	public void merge(MutableAggregationBuffer buff1, Row buff2) {
		TreeMap<Double, Double> internalBuffer = new TreeMap<>();
		// update in internal buffer
		internalBuffer.putAll((Map<Double, Double>) buff1.<Double, Double>getJavaMap(0));
		internalBuffer.putAll((Map<Double, Double>) buff2.<Double, Double>getJavaMap(0));

		// trim buffer to top N only this works best in distruibuted environment
		int bufferSize = internalBuffer.size();
		if (bufferSize > k) {
			for (int x = bufferSize; x > k; x--) {
				internalBuffer.remove(internalBuffer.lastKey());// trim the tree to keep top K or smallest K values
			}
		}
		//log.debug("merging...");
		buff1.update(0, internalBuffer);
	}

	@Override
	public Object evaluate(Row input) {
	  
		// in end do the contest
		Double frequentLabel = -1d;
		Map<Double, Double> map = input.getJavaMap(0);

		HashMap<Double, Integer> c = new HashMap<>(k);

		for (Double label : map.values()) {
			if (c.containsKey(label)) {
				c.put(label, c.get(label) + 1);// increment counter
			} else {
				c.put(label, 1);
			}
		}
		// we have counts now we will do the pass through to find top label frequency
		// TODO: include code snippet in report like evaluate(){.... ABC}
		int maxFreq = 0;
		for (Entry<Double, Integer> entry : c.entrySet()) {
			int freq = entry.getValue();
			if (freq > maxFreq) {
				maxFreq = freq;
				frequentLabel = entry.getKey();
			}
		}
		//log.info("computed prediction... " + frequentLabel);
		return frequentLabel;
	}
}