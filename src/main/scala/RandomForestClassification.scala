import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.{SparkConf, SparkContext}


object RandomForestClassification {

  def main(args: Array[String]) {

    val methods = new SfCrimeMethods

    val trainDataPath = "C:/Users/rahul/Documents/machineLearning/Kaggle/SFCrime/data/train.csv";
    val testDataPath = "C:/Users/rahul/Documents/machineLearning/Kaggle/SFCrime/data/test.csv";

    val sparkConf = new SparkConf().setAppName("MachineLearning")

    val sc = new SparkContext(sparkConf)

    val train = sc.textFile(trainDataPath).map(l => methods.parseTrainCsv(l))
    val test = sc.textFile(testDataPath).map(l => methods.parseTestCsv(l))

    val combinedData = train.union(test)

    val categories = combinedData.map(l => l.Category).distinct().filter(l => l != null).zipWithIndex().mapValues(_.toInt)
    val DayOfWeek = combinedData.map(l => l.DayOfWeek).distinct().zipWithIndex().mapValues(_.toInt + 1)
    val PdDistrict = combinedData.map(l => l.PdDistrict).distinct().zipWithIndex().mapValues(_.toInt + 1)

    categories.foreach(l => {

      println("Category: " + l._1 + " value: " + l._2)
    })

    val comCat = combinedData.keyBy(l => l.Category).join(categories).values
      .keyBy(s =>s._1.DayOfWeek).join(DayOfWeek).values
      .keyBy(s => s._1._1.PdDistrict ).join(PdDistrict).values.map(l => methods.sfCrime_sfCrimeNum(l))

    val trainNum = comCat.filter(s => s.Id == -1)
      .map(r=> LabeledPoint(r.Category, Vectors.dense(r.Address, r.DayOfWeek, r.Hour, r.Month, r.PdDistrict, r.X, r.Y)))

    val Array(trainData, cvData, testData) = trainNum.randomSplit(Array(0.8, 0.1, 0.1))

    trainData.cache()
    cvData.cache()
    testData.cache()

//    val testNum = comCat.filter(s => s.Id >= 0).sortBy(l => l.Id).map(r=> Vectors.dense(r.Address, r.DayOfWeek, r.Hour, r.Month, r.PdDistrict, r.X, r.Y))

    val numClasses = categories.count().toInt
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 60
    val featureSubsetStrategy = "auto"
    val impurity = "gini"
    val maxDepth = 8
    val maxBins = 32
    val seed = 1331

    val model = RandomForest.trainClassifier(trainNum, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity,
      maxDepth, maxBins, seed)

    //   testData.map(l => (model.predict(l.features), l.label)).foreach(l => println(l))

    val metrics = methods.getMetrics(model, cvData)

    println(metrics.confusionMatrix)

    println(metrics.precision)

    //    (0 until 39).map(cat => (metrics.precision(cat), metrics.recall(cat))
    //    ).foreach(println)

  }

}