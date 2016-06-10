import au.com.bytecode.opencsv.CSVParser
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD

class SfCrimeMethods extends Serializable{

  def getMetrics(model: RandomForestModel, data: RDD[LabeledPoint]): MulticlassMetrics = {

    val predictionsAndLabels = data.map(example =>
      (model.predict(example.features), example.label)
    )
    new MulticlassMetrics(predictionsAndLabels)
  }

  def getMetrics(model: GradientBoostedTreesModel, data: RDD[LabeledPoint]): MulticlassMetrics = {

    val predictionsAndLabels = data.map(example =>
      (model.predict(example.features), example.label)
    )
    new MulticlassMetrics(predictionsAndLabels)
  }


  def sfCrime_sfCrimeNum(l:(((sfCrime, Int), Int), Int)): sfCrimeNum ={

    val address = if (l._1._1._1.Address.toLowerCase.contains("block")) 1 else 0
    val year = l._1._1._1.Date.split(" ")(0).split("-")(0).toInt
    val month = l._1._1._1.Date.split(" ")(0).split("-")(1).toInt
    val hour = l._1._1._1.Date.split(" ")(1).split(":")(0).toInt

    new sfCrimeNum(
      l._1._1._1.Id,
      year,
      month,
      hour,
      l._1._1._2,
      l._1._2,
      l._2,
      address,
      X = l._1._1._1.X,
      Y = l._1._1._1.Y
    )
  }

  def parseTrainCsv(line: String): sfCrime ={

    val parser = new CSVParser(',')
    val parts = parser.parseLine(line)


    new sfCrime(
      -1,
      parts(0),
      parts(1),
      parts(3),
      parts(4),
      parts(6),
      parts(7).toDouble,
      parts(8).toDouble
    )

  }

  def parseTestCsv(line: String): sfCrime ={

    val parser = new CSVParser(',')
    val parts = parser.parseLine(line)

    new sfCrime(
      parts(0).toInt,
      parts(1),
      null,
      parts(2),
      parts(3),
      parts(4),
      parts(5).toDouble,
      parts(6).toDouble
    )

  }

}
