const { isUndefined, map, reduce } = require("lodash")
const { SupervisedLearner } = require("mlt-node")

const k = 50
const isClassification = true
const useDistanceWeight = true

class KnnLearner extends SupervisedLearner {
  train(features, labels) {
    features.normalize()
    this.features = features.m_data
    this.labels = labels.m_data
  }

  predict(features, labels) {
    const closestKInstances = this.getClosestKInstances(features)
    if (isClassification) {
      // Classification
      const mostCommonLabel = reduce(
        closestKInstances,
        (memo, inst) => {
          const l = this.labels[inst.i][0]
          const val = useDistanceWeight ? inst.weight : 1
          memo[l] = memo[l] ? memo[l] + val : val
          memo["_most"] = memo[l] > memo[memo._most] ? l : memo["_most"]
          return memo
        },
        { _start: 0, _most: "_start" }
      )["_most"]
      labels[0] = mostCommonLabel
    } else {
      // Regression
      let sum = 0
      for (let i = 0; i < closestKInstances.length; i++) {
        const valToAdd =
          (useDistanceWeight ? closestKInstances[i].weight : 1) *
          closestKInstances[i].distance
        sum += valToAdd
      }
      labels[0] = sum / k
    }
  }

  getClosestKInstances(features) {
    const distances = map(this.features, (instance, i) => {
      const distance = this.computeDistance(instance, features)
      return {
        i,
        distance,
        weight: distance !== 0 ? 1 / Math.pow(distance, 2) : 100000,
        instance
      }
    })
    distances.sort((a, b) => a.distance - b.distance)
    return distances.slice(0, k)
  }

  computeDistance(inst, features) {
    return Math.sqrt(
      reduce(
        inst,
        (memo, f, i) => {
          return memo + !isUndefined(f) && !isUndefined(features[i])
            ? Math.abs(f - features[i])
            : 1
        },
        0
      )
    )
  }
}

module.exports.KnnLearner = KnnLearner
