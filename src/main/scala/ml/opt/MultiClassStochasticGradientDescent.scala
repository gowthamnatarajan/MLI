package mli.ml.opt

import mli.interface._


case class MultiClassStochasticGradientDescentParameters(
    learningRate: Double = 1e-2,
    wInit: MLMatrix,
    maxIter: Int = 100,
    eps: Double = 1e-6,
    grad: (MLRow, MLMatrix) => (MLRow, MLRow)
  ) extends MLOptParameters
  
  

object MulticlassStochasticGradientDescent extends MLOpt with Serializable {

  def apply(data: MLTable, params: MultiClassStochasticGradientDescentParameters): MLMatrix = {
    runSGD(data, params.wInit, params.learningRate, params.grad, params.maxIter, params.eps)
  }
  
  def runSGD(
              data: MLTable,
              wInit: MLMatrix,
              learningRate: Double,
              grad: (MLRow, MLMatrix) => (MLRow, MLRow),
              maxIter: Int,
              eps: Double
            ): MLMatrix = {
       var weights = wInit
           val n = data.numRows
    var i = 0

    //Main loop of SGD. Calls local SGD and averages parameters. Checks for convergence after each pass.
    while(i < maxIter) {
      weights = data.matrixBatchMap(localSGD(_, weights, learningRate, grad)).reduce(_ plus _) over n

      i+=1
    }
      weights
  }
  
    /**
   * Locally runs SGD on each partition of data. Sends results back to master after each pass.
   */
  def localSGD(data: MLMatrix, weights: MLMatrix, lambda: Double, gradientFunction: (MLRow, MLMatrix) => (MLRow, MLRow)): MLMatrix = {
    null
  }
}