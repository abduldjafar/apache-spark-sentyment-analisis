// Databricks notebook source
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession._
import org.apache.spark.sql.{DataFrame,Column}
import org.apache.spark.ml.feature.{HashingTF, IDF, StopWordsRemover}
import org.apache.spark.ml.classification.{LogisticRegression}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator}
import java.util.regex.Pattern
import scala.collection.mutable.WrappedArray

// COMMAND ----------

val spark = builder.master("local").appName("Spark Sentyment Analysyst").getOrCreate;

// COMMAND ----------

// function for regex findall
def regexp_extractAll = udf((job: String, exp: String, groupIdx: Int) => {
    val pattern = Pattern.compile(exp.toString)
    val m = pattern.matcher(job.toString)
    var result = Seq[String]()
    while (m.find) {
      val temp =
        result =result:+m.group(groupIdx)
    }
    result.toArray
  })
  
 // function for check string
 def parseToInt = udf((entry: String) => {
     try {
        entry.toInt
      } catch {
        case e: Exception => 0
    }
     })

 def cleansingTweet = udf((entry: String) => {
     // Remove HTML special entities (e.g. &amp;)
     val regexhtml ="\\&\\w*;".r
     val result1 = regexhtml.replaceAllIn(entry,"")
     
     //remove @username
     val regexAd = "@[^\\s]+".r
     val result2 = regexAd.replaceAllIn(result1,"")
     
     //remove RT
     val regexRT = "rt".r
     val result3 = regexRT.replaceAllIn(result2,"")

     //remove tickers
     val regextTicks = "\\$\\w*".r
     val result4 = regextTicks.replaceAllIn(result3,"")
     
     //convert to lower
     val result5 = result4.toLowerCase()
     
     //remove hyperlinks
     val regexHyper = "https?:\\/\\/.*\\/\\w*".r
     val result6 = regexHyper.replaceAllIn(result5,"")
     
     //remove hashtag
     val regexHastag = "#\\w*".r
     val result7 = regexHastag.replaceAllIn(result6,"")
     
     result7.trim()
     
     }
 )
 // function for new dataframe
def cleanDf()(df: DataFrame ): DataFrame = {
 (
      df.select("_c0")
        .withColumn("new",split($"_c0",";")).select($"new".getItem(0).as("tweet"),$"new".getItem(1).as("label"))
        .withColumn("label_temp",parseToInt($"label"))
        .select("tweet","label_temp").withColumnRenamed("label_temp","label")
  )
}

// COMMAND ----------

val df_temp = spark.read.format("csv").option("delimeter",";").load("dataset/training_all*")
val stopword =  spark.read.format("csv").load("stopword*")
val df = df_temp.transform(cleanDf())
val listStopWord = stopword.select("_c0").collect().map(_(0)).toArray.map(_.asInstanceOf[String])

// COMMAND ----------

val data = (
    df.distinct.select("tweet","label")
    .withColumn("cleanedWord", cleansingTweet(new Column("tweet")))
    .withColumn("countwords",size(regexp_extractAll(new Column("cleanedWord"),lit("\\w+"),lit(0))))
    .filter($"countwords" >= "2")
    )

// COMMAND ----------

val dataFreqword = (
    data.select("cleanedWord")
        .withColumn("SplittedWord",regexp_extractAll(new Column("cleanedWord"),lit("\\w+"),lit(0)) )
        .withColumn("WordCount",explode(col("SplittedWord")))
        .select("WordCount")
        .groupBy("WordCount").count()
        .withColumnRenamed("WordCount","words")
        .withColumnRenamed("count","total")
        .orderBy(desc("total"))
    )

// COMMAND ----------

val dataWithList = (
        data.withColumn("ListWord",regexp_extractAll(new Column("cleanedWord"),lit("\\w+"),lit(0)) )
    )

val remover = (
    new StopWordsRemover()
    .setStopWords(listStopWord)
    .setInputCol("ListWord")
    .setOutputCol("filteredWithStopWord")
    )
val dataAfterStop = remover.transform(dataWithList)

// COMMAND ----------

// Prepare data before training

// make TF
val hashingTF = (
    new HashingTF()
  .setInputCol("filteredWithStopWord").setOutputCol("rawFeatures").setNumFeatures(Math.pow(2.0,16.0).toInt)
  )

val featurizedData = hashingTF.transform(dataAfterStop)

// create TF-IDF model
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val idfModel = idf.fit(featurizedData)


val rescaledData = idfModel.transform(featurizedData)
val rescaledDatas = rescaledData.select("ListWord","rawFeatures","features","label")


val splits = rescaledDatas.randomSplit(Array(0.6, 0.4), seed = 11L)
val training_data = splits(0).cache()
val test_data = splits(1)


// COMMAND ----------

// train model
val model = new LogisticRegression().fit(training_data)

//prediction model
val predictions = model.transform(test_data)

// COMMAND ----------

val evaluator = new BinaryClassificationEvaluator("prediction")
evaluator.evaluate(predictions)
