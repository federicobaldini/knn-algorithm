import { tensor } from "@tensorflow/tfjs";
import { knn } from "./algorithms/k-nearest-neighbors";
import { loadCSV } from "./csv-loader";
import { LinearRegression } from "./algorithms/linear-regression";

type Dataset = Array<Array<string | number>>;

const runKnnAnalysis = async (): Promise<void> => {
  let {
    features, // input features for training data.
    labels, // output labels for training data.
    testFeatures, // input features for test data.
    testLabels, // output labels for test data.
  }: {
    features: Dataset;
    labels: Dataset;
    testFeatures?: Dataset;
    testLabels?: Dataset;
  } = loadCSV("house.csv", {
    datasetColumns: ["lat", "long", "sqft_lot", "sqft_living"],
    labelColumns: ["price"],
    shuffle: true,
    splitTest: 10,
  });

  // If there is test data.
  if (testFeatures !== undefined) {
    // Iterate through each row of test features.
    testFeatures.forEach(
      (testFeature: Array<string | number>, index: number): void => {
        // Use the k-nearest neighbor algorithm to predict output label for test data.
        const result: number = knn(
          tensor(features), // training input features.
          tensor(labels), // training output labels.
          tensor(testFeature), // test input features.
          10 // number of nearest neighbors to consider.
        );
        // If there are test labels.
        if (testLabels !== undefined) {
          // Calculate the percentage error between predicted and actual labels.
          const error: number =
            ((testLabels[index][0] as number) - result) /
            (testLabels[index][0] as number);
          // Log the percentage error to the console
          console.log(
            `Error for feature ${index}: ${(error * 100).toFixed(2)}%`
          );
        }
      }
    );
  }
};

const runLinearRegressionAnalysis = async (): Promise<void> => {
  let {
    features, // input features for training data.
    labels, // output labels for training data.
    testFeatures, // input features for test data.
    testLabels, // output labels for test data.
  }: // testFeatures, // input features for test data.
  // testLabels, // output labels for test data.
  {
    features: Dataset;
    labels: Dataset;
    testFeatures?: Dataset;
    testLabels?: Dataset;
  } = loadCSV("cars.csv", {
    datasetColumns: ["horsepower"],
    labelColumns: ["mpg"],
    shuffle: true,
    splitTest: 50,
  });

  const linearRegression: LinearRegression = new LinearRegression(
    tensor(features),
    tensor(labels),
    { learningRate: 0.0001, iterations: 100 }
  );

  linearRegression.train();

  console.log("Updated M is:", linearRegression.getM());
  console.log("Updated B is:", linearRegression.getB());

  if (testFeatures && testLabels) {
    linearRegression.test(tensor(testFeatures));
  }
};

export { runKnnAnalysis, runLinearRegressionAnalysis };
