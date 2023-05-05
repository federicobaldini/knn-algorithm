/**
 * K-Nearest Neighbor algorithm implementation in TypeScript.
 */

import { chain, shuffle, last, initial, cloneDeep, min, max } from "lodash";

type Dataset = Array<Array<number>>;

/**
 * This function scales the values of a dataset to a value between 0 and 1 based on
 * the minimum and maximum values for each feature column.
 *
 * @param dataset The dataset to be scaled.
 * @param featureCount The number of features in the dataset.
 * @returns The scaled dataset.
 */
const minMax = (dataset: Dataset, featureCount: number): Dataset => {
  // Create a deep clone of the dataset to avoid modifying the original dataset.
  const clonedDataset: Dataset = cloneDeep(dataset);

  // Iterate over each feature column in the dataset.
  for (
    let columnIndex: number = 0;
    columnIndex < featureCount;
    columnIndex += 1
  ) {
    // Extract the column data for the current feature.
    const columnData: Array<number> = clonedDataset.map(
      (rowData: Array<number>) => rowData[columnIndex]
    );
    // Calculate the minimum and maximum values for the feature.
    const minValue: number | undefined = min(columnData);
    const maxValue: number | undefined = max(columnData);

    // If both minimum and maximum values are defined, scale the feature values for each data point in the column.
    if (maxValue !== undefined && minValue !== undefined) {
      for (
        let rowIndex: number = 0;
        rowIndex < clonedDataset.length;
        rowIndex += 1
      ) {
        clonedDataset[rowIndex][columnIndex] =
          (clonedDataset[rowIndex][columnIndex] - minValue) /
          (maxValue - minValue);
      }
    }
  }

  return clonedDataset;
};

/**
 * This function calculates the n-dimensional distance between two points using
 * the Pythagorean theorem for n-dimensional triangles.
 *
 * @param pointsA An array representing the coordinates of the first point.
 * @param pointsB An array representing the coordinates of the second point.
 * @returns The distance between the two points as a number.
 * @throws An error if pointsA and pointsB do not have the same size.
 */
const nDimensionalDistance = (
  pointsA: Array<number>,
  pointsB: Array<number>
): number | undefined => {
  if (pointsA.length === pointsB.length) {
    /*
      - ZIP:
        Creates an array of pairs of corresponding elements from the two input arrays (pointsA and pointsB).
        Example: [ [ 1, 2, 3 ], [ 4, 5, 6 ] ] becomes [ [ 1, 4 ], [ 2, 5 ], [ 3, 6 ] ]

      - MAP:
        Calculates the square of the difference between the elements of each pair in the array of pairs
        created by the "zip" function.
        Example: [ [ 1, 4 ], [ 2, 5 ], [ 3, 6 ] ] becomes [ 9, 9, 9 ]

      - SUM:
        Sum the squared differences of the pairs obtained from the "map" function,
        and then calculate the square root of the result which represents the n-dimensional distance between the two points.
        Example: 27
    */
    return (
      chain(pointsA)
        .zip(pointsB)
        .map(([a, b]: [number, number]) => (a - b) ** 2)
        .sum()
        .value() ** 0.5
    );
  }
  throw new Error(
    "nDimensionalDistance: the two points array must have the same size."
  );
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
 * @param predictionPoints The points used to calculate the distance between each training data.
 * @param k k-nearest neighbors.
 * @returns The predicted label for the test point.
 */
const knn = (
  dataset: Dataset,
  predictionPoints: Array<number>,
  k: number
): number => {
  /*
    - MAP:
      Calculates the distance between each training data point and the prediction point(s),
      and maps each training data point to a new array that pairs its distance with its dependent variable value.
      Returns an array of pairs [distance, dependent variable].
      Example: [ [ 5, 2 ], [ 15, 3 ], [ 20, 2 ] ]

    - SORT_BY:
      Sorts the array of pairs by distance in ascending order.
      Example: [ [ 5, 2 ], [ 15, 3 ], [ 20, 2 ] ]

    - SLICE:
      Selects the k elements (i.e. the k-nearest neighbors) from the sorted array.
      Example: [ [ 5, 2 ], [ 15, 3 ], [ 20, 2 ] ] (with k = 2) becomes [ [ 5, 2 ], [ 15, 3 ] ]

    - COUNT_BY:
      Groups the k-nearest neighbors by their dependent variable value, and counts the occurrences of each value.
      Returns an object whose keys are the unique dependent variable values, and whose values are the frequencies
      of those values among the k-nearest neighbors.
      Example: { "2": 2, "3": 1 }

    - TO_PAIRS:
      Converts the object returned by countBy to an array of pairs [dependent variable, frequency].
      Example: [ [ "3", "1" ], [ "2", "2" ] ]

    - SORT_BY:
      Sorts the array of pairs by frequency in descending order.
      Example: [ [ "2", "2" ], [ "3", "1" ] ]

    - LAST:
      Takes the last (i.e. highest frequency) pair from the sorted array.
      Example: [ "3", "1" ] (when there are ties, this function will pick one at random)

    - FIRST:
      Takes the first element (i.e. the dependent variable value) from the last pair.
      Example: "3"

    - PARSE_INT:
      Converts the string value of the dependent variable to a number.
      Example: 3
  */
  return chain(dataset)
    .map((data: Array<number>) => [
      nDimensionalDistance(initial(data), predictionPoints),
      last(data),
    ])
    .sortBy((data: Array<number | undefined>) => data[0])
    .slice(0, k)
    .countBy((data: Array<number | undefined>) => data[1])
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
  const featuresList: Array<number> = [0, 1, 2];
  const k: number = 10;

  featuresList.forEach((feature: number) => {
    const filteredDatasetByFeature: Dataset = dataset.map(
      (data: Array<number>) => [data[feature], last(data) as number]
    );
    const [testDataset, trainingDataset]: [Dataset, Dataset] = splitDataset(
      minMax(filteredDatasetByFeature, 1),
      testDatasetSize
    );

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
          knn(trainingDataset, initial(testData), k) === last(testData)
      )
      .size()
      .divide(testDatasetSize)
      .value();

    console.log(
      "feature: ",
      feature,
      " accuracy: " + Math.round(accuracy * 100) + "%"
    );
  });
};

export { analyzeDataset };
