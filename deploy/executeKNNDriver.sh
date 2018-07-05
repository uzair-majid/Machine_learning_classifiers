#!/bin/bash
 
 

spark-submit --class a2.DriverApp   --master yarn  --jars breeze-macros_2.11-0.13.2.jar,breeze_2.11-0.13.2.jar --deploy-mode client  --conf spark.local.dir=/home/asye5989/lab1   --files KNN_Driver.properties  Assignment-2.
