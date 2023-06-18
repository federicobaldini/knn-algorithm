import { tensor } from "@tensorflow/tfjs";
import { knn } from "./algorithms/k-nearest-neighbors";
import { loadCSV } from "./csv-loader";
import { LinearRegression } from "./algorithms/linear-regression";
import { BinaryLogisticRegression } from "./algorithms/binary-logistic-regression";
import { MultinominalLogisticRegression } from "./algorithms/multinominal-logistic-regression";
import { flatMap } from "lodash";
import { training } from "mnist-data";

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
  }: {
    features: Dataset;
    labels: Dataset;
    testFeatures?: Dataset;
    testLabels?: Dataset;
  } = loadCSV("cars.csv", {
    datasetColumns: ["horsepower", "weight", "displacement"],
    labelColumns: ["mpg"],
    shuffle: true,
    splitTest: 50,
  });

  const linearRegression: LinearRegression = new LinearRegression(
    tensor(features),
    tensor(labels),
    { learningRate: 0.1, iterations: 3, batchSize: 10 }
  );

  linearRegression.train();

  if (testFeatures && testLabels) {
    linearRegression.test(tensor(testFeatures), tensor(testLabels));
  }

  linearRegression.predict(tensor([[120, 2, 380]])).print();
};

const runBinaryLogisticRegressionAnalysis = async (): Promise<void> => {
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
  } = loadCSV("cars.csv", {
    datasetColumns: ["horsepower", "weight", "displacement"],
    labelColumns: ["passedemissions"],
    shuffle: true,
    splitTest: 50,
    converters: {
      passedemissions: (value: string): number => {
        return value === "TRUE" ? 1 : 0;
      },
    },
  });

  const binaryLogisticRegression: BinaryLogisticRegression =
    new BinaryLogisticRegression(tensor(features), tensor(labels), {
      learningRate: 0.5,
      iterations: 100,
      batchSize: 10,
      decisionBoundary: 0.6,
    });

  binaryLogisticRegression.train();

  if (testFeatures && testLabels) {
    binaryLogisticRegression.test(tensor(testFeatures), tensor(testLabels));
  }

  binaryLogisticRegression.predict(tensor([[88, 1.065, 127]])).print();
  binaryLogisticRegression.predict(tensor([[120, 2, 380]])).print();
};

const runMultinominalLogisticRegressionAnalysis = async (): Promise<void> => {
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
  } = loadCSV("cars.csv", {
    datasetColumns: ["horsepower", "weight", "displacement"],
    labelColumns: ["mpg"],
    shuffle: true,
    splitTest: 50,
    converters: {
      mpg: (value: string): Array<number> => {
        const mpg: number = parseFloat(value);
        if (mpg < 15.0) {
          return [1, 0, 0];
        }
        if (mpg < 30.0) {
          return [0, 1, 0];
        }
        return [0, 0, 1];
      },
    },
  });

  const multinominalLogisticRegression: MultinominalLogisticRegression =
    new MultinominalLogisticRegression(
      tensor(features),
      tensor(flatMap(labels)),
      {
        learningRate: 0.5,
        iterations: 100,
        batchSize: 10,
      }
    );

  multinominalLogisticRegression.train();

  if (testFeatures && testLabels) {
    console.log(
      multinominalLogisticRegression.test(
        tensor(testFeatures),
        tensor(flatMap(testLabels))
      )
    );
  }

  multinominalLogisticRegression.predict(tensor([[150, 2.223, 200]])).print();
};

const runImageRecognition = (): void => {
  const mnistData = training(0, 1);
  console.log(mnistData.images.values);
};

export {
  runKnnAnalysis,
  runLinearRegressionAnalysis,
  runBinaryLogisticRegressionAnalysis,
  runMultinominalLogisticRegressionAnalysis,
  runImageRecognition,
};
