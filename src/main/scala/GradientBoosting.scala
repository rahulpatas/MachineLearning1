import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.{Algo, BoostingStrategy}
import org.apache.spark.{SparkConf, SparkContext}

object GradientBoosting {

  def main(args: Array[String]) {

    val methods = new SfCrimeMethods

    val trainDataPath = "C:/Users/rahul/Documents/machineLearning/Kaggle/SFCrime/data/train.csv";
    val testDataPath = "C:/Users/rahul/Documents/machineLearning/Kaggle/SFCrime/data/test.csv";

    val sparkConf = new SparkConf().setAppName("MachineLearning")

    val sc = new SparkContext(sparkConf)

    val train = sc.textFile(trainDataPath).map(l => methods.parseTrainCsv(l))
    val test = sc.textFile(testDataPath).map(l => methods.parseTestCsv(l))

    val combinedData = train.union(test)

    println(combinedData.map(l => l.Category).distinct().count())
    println(combinedData.map(l => l.PdDistrict).distinct().count())

    val categories = combinedData.map(l => l.Category).filter(l => l != null).distinct().zipWithIndex().mapValues(_.toInt + 1);
    val DayOfWeek = combinedData.map(l => l.DayOfWeek).distinct().zipWithIndex().mapValues(_.toInt + 1);
    val PdDistrict = combinedData.map(l => l.PdDistrict).distinct().zipWithIndex().mapValues(_.toInt + 1);

    categories.foreach(l => {

      println("Category: " + l._1 + " value: " + l._2)
    })

    val comCat = combinedData.keyBy(l => l.Category).join(categories).values
      .keyBy(s =>s._1.DayOfWeek).join(DayOfWeek).values
      .keyBy(s => s._1._1.PdDistrict ).join(PdDistrict).values.map(l => methods.sfCrime_sfCrimeNum(l))


    val trainNum = comCat.filter(s => s.Id == -1)
      .map(r=> LabeledPoint(r.Category,
        Vectors.dense(r.Address, r.DayOfWeek, r.Hour, r.Month, r.PdDistrict, r.X, r.Y)))


    val testNum = comCat.filter(s => s.Id >= 0).map(r=> LabeledPoint(r.Id.toDouble,
      Vectors.dense(r.Address, r.DayOfWeek, r.Hour, r.Month, r.PdDistrict, r.X, r.Y))).sortBy(l => l.label)

    val Array(trainData, cvData, testData) = trainNum.randomSplit(Array(0.8, 0.1, 0.1))

    trainData.cache()
    cvData.cache()
    testData.cache()

    val boostingStrategy = BoostingStrategy.defaultParams(Algo.Regression)

    boostingStrategy.setNumIterations(100)
    boostingStrategy.setLearningRate(0.02)
    boostingStrategy.treeStrategy.setMaxDepth(4)
    boostingStrategy.treeStrategy.setNumClasses(categories.count().toInt)
    boostingStrategy.treeStrategy.setSubsamplingRate(0.7)

    val model = GradientBoostedTrees.train(trainNum, boostingStrategy)

    val metrics = methods.getMetrics(model, cvData)

    println(metrics.confusionMatrix)

    println(metrics.precision)

//    val testErr = testNum.map(l => model.predict(l.features)).foreach(l => println(l))

  }
}