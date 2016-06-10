import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

class XgBoost {

  def main (args: Array[String]) {
    val sparkConf = new SparkConf()
      .setAppName("MachineLearning")
//      .set("spark.serializer", "org.apache.spark.serializer.KyroSerializer")



    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    //
    //    val df = sqlContext.read
    //      .format("com.databricks.spark.csv")
    //      .option("header", "true") // Use first line of all files as header
    //      .option("inferSchema", "true") // Automatically infer data types
    //      .load(dataPath)
    //
    //    df.select("Dates").map(date => {
    //      val format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
    //     println( format.parse(date.getString(0)).getTime)
    //    })

    val start = System.currentTimeMillis()
    println(sc.textFile("C:/Users/rahul/Documents/machineLearning/Kaggle/SFCrime/train.csv").count())
    println(s"Total time: ${System.currentTimeMillis() - start} ms")
  }


}
