const { SupervisedLearner, BaselineLearner, run } = require("mlt-node")

const { KnnLearner } = require("./knn/knnLearner")

function getLearner(model) {
  switch (model) {
    case "baseline":
      return new BaselineLearner()
    case "perceptron":
    // return new Perceptron();
    case "neuralnet":
    // return new NeuralNet();
    case "decisiontree":
    // return new DecisionTree();
    case "knn":
      return new KnnLearner()
    default:
      throw new Error("Unrecognized model: " + model)
  }
}

//Parse the command line arguments
const learnerName = process.argv[3]

run(getLearner(learnerName))
