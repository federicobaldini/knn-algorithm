/**
 * K-Nearest Neighbor algorithm implementation in TypeScript.
 * (using a single indipendent variable)
 */

import { chain, shuffle } from "lodash";

type Dataset = Array<Array<number>>;

const distance = (pointA: number, pointB: number): number => {
  return Math.abs(pointA - pointB);
};

const splitDataset = (
  dataset: Dataset,
  testDataCount: number
): [Dataset, Dataset] => {
  const shuffledDataset: Dataset = shuffle(dataset);
  const testDataset: Dataset = shuffledDataset.slice(0, testDataCount);
  const trainingDataset: Dataset = shuffledDataset.slice(testDataCount);
  return [testDataset, trainingDataset];
};

/**
 * Analyzes a dataset using the k-nearest neighbors algorithm.
 *
 * Given a set of training data and a test point, this function predicts
 * the label of the test point based on the labels of its k-nearest neighbors.
 *
 * The dataset is an array of arrays, where each subarray represents a training data point.
 * Each training data point in the dataset should be an array of numbers:
 *   0: independent variable (position on the x-axis)
 *   1: dependent variable (label)
 *
 * The function works by:
 * 1. Computing the Euclidean distance from the test point to each data point in the dataset.
 * 2. Selecting the k nearest neighbors.
 * 3. Counting the occurrences of each label among the k nearest neighbors.
 * 4. Returning the most common label (i.e. the label with the highest count).
 *
 * The algorithm uses the "bucket" variable (data[3]) as the dependent variable that is analyzed.
 *
 * @param dataset The set of training data to be analyzed.
 * @param predictionPoint The point used to calculate the distance between each training data.
 * @param k k-nearest neighbors.
 * @returns The predicted label for the test point.
 */
const knn = (dataset: Dataset, predictionPoint: number, k: number): number => {
  /*
    - MAP:
      Calculates the distance between each training data point and the prediction point,
      and maps each training data point to a new array that pairs its distance with its "bucket" value.
      Returns an array of pairs [distance, bucket].
      Example: [ [ 5, 2 ], [ 15, 3 ], [ 20, 2 ] ]

    - SORT_BY:
      Sorts the array of pairs by distance in ascending order.
      Example: [ [ 5, 2 ], [ 15, 3 ], [ 20, 2 ] ]

    - SLICE:
      Selects the k elements (i.e. the k-nearest neighbors) from the sorted array.
      Example: [ [ 5, 2 ], [ 15, 3 ], [ 20, 2 ] ] (with k = 2) becomes [ [ 5, 2 ], [ 15, 3 ] ]

    - COUNT_BY:
      Groups the k-nearest neighbors by their "bucket" value, and counts the occurrences of each bucket value.
      Returns an object whose keys are the unique bucket values, and whose values are the frequencies
      of those bucket values among the k-nearest neighbors.
      Example: { "2": 2, "3": 1 }

    - TO_PAIRS:
      Converts the object returned by countBy to an array of pairs [bucket, frequency], in ascending order of frequency.
      Example: [ [ "3", 1 ], [ "2", 2 ] ]

    - SORT_BY:
      Sorts the array of pairs by frequency in descending order.
      Example: [ [ "2", 2 ], [ "3", 1 ] ]

    - LAST:
      Takes the last (i.e. highest frequency) pair from the sorted array.
      Example: [ "2", 2 ] (when there are ties, this function will pick one at random)

    - FIRST:
      Takes the first element (i.e. the bucket value) from the last pair.
      Example: "2"

    - PARSE_INT:
      Converts the string value of the bucket to a number.
      Example: 2
  */
  return chain(dataset)
    .map((data: Array<number>) => [distance(data[0], predictionPoint), data[3]])
    .sortBy((data: Array<number>) => data[0])
    .slice(0, k)
    .countBy((data: Array<number>) => data[1])
    .toPairs()
    .sortBy((data: Array<string | number>) => data[1])
    .last()
    .first()
    .parseInt()
    .value();
};

/**
 * Analyzes a dataset and calculates the accuracy of the K-Nearest Neighbor algorithm.
 *
 * Given a dataset, this function splits the data into a test dataset and a training dataset.
 * It then loops through each test point in the test dataset, using the K-Nearest Neighbor algorithm
 * (implemented in the knn function) to predict the label of the test point based on the labels of its k-nearest neighbors
 * from the training dataset. It compares the predicted label with the actual label for the test point and
 * calculates the accuracy of the algorithm as the ratio of the number of correctly predicted test points
 * to the total number of test points.
 *
 * The dataset is an array of arrays, where each subarray represents a training or test data point.
 * Each training or test data point in the dataset should be an array of numbers:
 * 0: independent variable (position on the x-axis)
 * 1: dependent variable (label)
 *
 * The K-Nearest Neighbor algorithm uses the "bucket" variable (data[3]) as the dependent variable that is analyzed.
 *
 * @param dataset The set of data to be analyzed.
 * @returns Nothing - the function logs the accuracy to the console.
 */
const analyzeDataset = (dataset: Dataset): void => {
  const testDatasetSize: number = 100;
  const [testDataset, trainingDataset]: [Dataset, Dataset] = splitDataset(
    dataset,
    testDatasetSize
  );

  for (let k = 1; k < 20; k += 1) {
    /*
      - FILTER:
        Filters the test dataset by selecting only the test data points for which
        the predicted label matches the actual label. Uses the knn() function to predict the label.

      - SIZE:
        Calculates the number of correctly predicted test data points
        (i.e. the number of elements in the filtered array).

      - DIVIDE:
        Divides the number of correctly predicted test data points by the total number
        of test data points to calculate the accuracy of the K-Nearest Neighbor algorithm.
    */
    const accuracy: number = chain(testDataset)
      .filter(
        (testData: Array<number>) =>
          knn(trainingDataset, testData[0], k) === testData[3]
      )
      .size()
      .divide(testDatasetSize)
      .value();

    console.log("k: ", k, " accuracy: " + Math.round(accuracy * 100) + "%");
  }
};

export { analyzeDataset };
