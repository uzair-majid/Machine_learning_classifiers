package a2;

 
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class NaiveBayesClassification {

	public static void main(String[] args) { 
		  SparkSession	spark = SparkSession.builder().master("yarn").appName("Java Spark Machine Learning NaiveBayes").getOrCreate();
			
		  // String inputDataPath = "/home/uzair/Desktop/Shared_VM/data/"; //input path with for files
		  // String inputDataPath = "/home/uzair/Desktop/Shared_VM/data/"; //input path with for files

	      // String trainingSet = inputDataPath + "Train-label-28x28.csv";
	       //String testSet = inputDataPath + "Test-label-28x28.csv";	
	       String trainingSet = args[0];
	       String	testSet = args[1];
	    		   
	       Dataset<Row> df = spark.read().option("header", "false").option("inferSchema",true).csv(trainingSet); //training 
	       Dataset<Row> df2 = spark.read().option("header", "false").option("inferSchema",true).csv(testSet);//test
	       //Dataset<Row> traininglabel = spark.read().option("header", "false").option("inferSchema",true).csv(trainingLabel);//test
	      // Dataset<Row> testlabel = spark.read().option("header", "false").option("inferSchema",true).csv(testLabel);//test
           
	     //  if (args.length >= 2) {


		//	}

	       df=df.withColumnRenamed("_c0", "label");

	       df2=df2.withColumnRenamed("_c0", "label");

	       //scala.collection.immutable.List<StructField> Schemalist= df.schema().toList();
	         //Schemalist.drop(0);
	       
	          
	        		  
	        		String temp[] = df.schema().fieldNames(); 
	        		 List<String> templist= Arrays.asList(temp);
	        		 
	        		
	        		String[] featureslist=templist.subList(1, templist.size()).toArray(new String[0]);

	        		//for(int i=0;i<featureslist.length;i++) {
	        			//updatedschema[i]= Schemalist.take(i).toString();
	        			//System.out.println(featureslist[i]);
	        		//}
	         
             
		    //traininglabel= traininglabel.toDF("label2");

	       System.out.println("++++++++++++++++++++++++++"+df.count()+"+++++++++++++++++++++++++++++++++++++++");
	       
	       VectorAssembler assembler = new VectorAssembler()
	    		   .setInputCols(featureslist)
	    		   .setOutputCol("features");    
	      
//	       VectorAssembler traininglabelassembler = new VectorAssembler()
//	    		   .setInputCols(traininglabel.columns())
//	    		   .setOutputCol("label");
//			traininglabel = traininglabelassembler.transform(traininglabel).drop("label2").select("label");

//	       VectorAssembler testassembler = new VectorAssembler()
//	    		   .setInputCols(updatedschema)
//	    		   .setOutputCol("features");    
	       	
	       
		Dataset<Row> output = assembler.transform(df).select("label","features") ;
		Dataset<Row> output2 = assembler.transform(df2);

		//output.show(2);
		
		
		//PCAModel model = new PCA().setK(55).setInputCol("Features2").setOutputCol("features").fit(output);
		//Dataset<Row> trainingset = model.transform(output).select("label", "features");
		//trainingset.show(2);
		//Dataset<Row> testset = model.transform(output2).select("features");
//		StructType TrainingSetAndLabel = new StructType()
	//			.add("label","int")
		//		.add("features","string");
		
		// traininglabel.select("label").show(2);
		//PCAModel model2 = new PCA().setK(55).setInputCol("features").setOutputCol("pcaFeatures").fit(output2);
		
		//traininglabel.show(2);
		
		// trainingset = trainingset.union(traininglabel);
 //		
		NaiveBayes nb = new NaiveBayes();
////
////		// train the model
		NaiveBayesModel nbmodel = nb.fit(output);
//		
		Dataset<Row> predictions = nbmodel.transform(output2);
//	    predictions  = predictions.withColumn("labeltmp", predictions.col("label").cast(DoubleType))
//	    	    .drop("labeltmp")
//	    	    .withColumnRenamed("labeltmp", "label");
	    predictions.select("label","prediction").show(2);
//	    
	    MulticlassMetrics metrics = new MulticlassMetrics(predictions.selectExpr("cast(label as double) label","prediction"));

	 // Confusion matrix
	 org.apache.spark.mllib.linalg.Matrix confusion = metrics.confusionMatrix();
	// System.out.println("Confusion matrix: \n" + confusion);

	 // Overall statistics
	 System.out.println("Accuracy = " + metrics.accuracy());
//
	 // Stats by labels
	 for (int i = 0; i < metrics.labels().length; i++) {
	  // System.out.format("Class %f precision = %f\n", metrics.labels()[i]));
	   //System.out.format("Class %f recall = %f\n" );
	   System.out.println("label: "+metrics.labels()[i]+" precision: "+metrics.precision(metrics.labels()[i])+" recall: "+ metrics.recall(metrics.labels()[i])+" f1_Score "+ metrics.fMeasure(metrics.labels()[i]));
	 }
	 System.out.println();
	 System.out.println("Accuracy = " + metrics.accuracy());
	 }

} 