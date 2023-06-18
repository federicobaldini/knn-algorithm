import { Tensor, Rank, moments } from "@tensorflow/tfjs";

/**
 * Analyzes a dataset using the k-nearest neighbors algorithm.
 *
 * Given a set of training data and a test point, this function predicts
 * the label of the test point based on the labels of its k-nearest neighbors.
 *
 * The dataset is an array of arrays, where each subarray represents a training data point.
 * Each training data point in the dataset should be an array of numbers:
 *   0: independent variable
 *   1: dependent variable (label)
 *
 * The function works by:
 * 1. Computing the Euclidean distance from the test point to each data point in the dataset.
 * 2. Selecting the k nearest neighbors.
 * 3. Computing the average label of the k nearest neighbors.
 * 4. Returning the average label.
 *
 * @param features The set of training data to be analyzed (features).
 * @param labels The labels of the training data points (dependent variables).
 * @param predictionPoints The points used to calculate the distance between each training data.
 * @param k k-nearest neighbors.
 * @returns The predicted label for the test point.
 */
const knn = (
  features: Tensor<Rank.R2>,
  labels: Tensor<Rank.R2>,
  predictionPoints: Tensor<Rank.R2>,
  k: number
): number => {
  const { mean, variance }: { mean: Tensor<Rank>; variance: Tensor<Rank> } =
    moments(features, 0);

  features.print();

  const scaledPredictionPoints: Tensor<Rank.R2> = predictionPoints
    .sub(mean)
    .div(variance.pow(0.5));

  return (
    features
      // Values standardized to normal (Gaussian) distribution.
      .sub(mean)
      .div(variance.pow(0.5))
      // Find distances between the features and the prediction point.
      .sub(scaledPredictionPoints)
      .pow(2)
      .sum(1)
      .pow(0.5)
      // This step is used to preserve the relation between the distances and the labels.
      .expandDims(1)
      .concat(labels, 1)
      // Sort the distances in ascending order.
      .unstack()
      .map((t: Tensor<Rank>): Array<number> => t.arraySync() as Array<number>)
      .sort((a: Array<number>, b: Array<number>): 1 | -1 =>
        a[0] > b[0] ? 1 : -1
      )
      // Take the top k-nearest neighbors.
      .slice(0, k)
      // Compute the average label of the k-nearest neighbors.
      .reduce(
        (accumulator: number, pair: Array<number>): number =>
          accumulator + pair[1],
        0
      ) / k
  );
};

export { knn };
