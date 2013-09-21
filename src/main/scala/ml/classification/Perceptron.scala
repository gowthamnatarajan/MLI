package mli.ml.classification

import mli.ml._
import scala.math
import mli.interface._

class PerceptronModel(
    trainingTbl: MLTable,
    trainingParams: PerceptronParameters,
    trainingTime: Long,
    val weights: MLRow) 
    extends Model[PerceptronParameters](trainingTbl, trainingTime, trainingParams) {

    /** Predicts the label of a given data point. */
    def predict(x: MLRow) : MLValue = {
        null
    }

    /**
     * Provides a user-friendly explanation of this model.
     * For example, plots or console output.
     */
    def explain() : String = {
        "Weights: " + weights.toString
    }
}

case class PerceptronParameters(
    learningRate: Double = 0.2,
    maxIterations: Int = 100,
    minLossDelta: Double = 1e-5,
    numClass: Int = 2)
    extends AlgorithmParameters


object PerceptronAlgorithm extends Algorithm[PerceptronParameters] {

  def defaultParameters() = PerceptronParameters()
    
  def train(data: MLTable, params: PerceptronParameters): PerceptronModel = {
      
      val d = data.numCols - 1
      
    def gradient(row: MLRow, w: MLRow): MLRow = {
      val x = MLVector(row.slice(1,row.length))
      val y = row(0).toNumber
      var pred = Array(0)
      for(i <- w.toMLRows) {
        pred :+ (row dot i)
      }
      val predVal = pred
      val maxIndex = predVal.indexOf(predVal.max)
      if(maxIndex == y) {
        (null, null)
      }
      else {
        var j = 0
        var classWeight: MLRow = null
        var predictWeight: MLRow = null
        for(i <- w.toMLRows) {
          j = j + 1
          if(j == y) {
            classWeight = i
          }
          if(j == maxIndex) {
            predictWeight = i
          }
        }
        (classWeight, predictWeight)
      }
    }
      val startTime = System.currentTimeMillis
      val optParams = opt.StochasticGradientDescentParameters(wInit = MLMatrix.zeros(params.numClass, d), grad = gradient, 
          learningRate = params.learningRate)
      val weights = opt.StochasticGradientDescent(data, optParams)
      val trainTime =  System.currentTimeMillis - startTime
      new PerceptronModel(data, params, trainTime, weights)
  }
    
    def main(args: Array[String]): Unit = {}

}