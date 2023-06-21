import { tensor } from "@tensorflow/tfjs";
import { knn } from "./algorithms/k-nearest-neighbors";
import { loadCSV } from "./csv-loader";
import { LinearRegression } from "./algorithms/linear-regression";
import { BinaryLogisticRegression } from "./algorithms/binary-logistic-regression";
import { MultinominalLogisticRegression } from "./algorithms/multinominal-logistic-regression";
import { flatMap } from "lodash";
import mnist from "mnist-data";

type Dataset = Array<Array<string | number>>;

/**
 * Run k-nearest neighbors analysis.
 *
 * Predicts output labels for test data using the k-nearest neighbor algorithm.
 * Calculates the percentage error between predicted and actual labels (if available).
 */
const runKnnAnalysis = async (): Promise<void> => {
  // Load CSV file "house.csv" containing training and test data
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
      /**
       * Predict the output label for each test feature using k-nearest neighbors algorithm.
       *
       * @param testFeature - The test input feature.
       * @param index - The index of the test feature.
       */
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

/**
 * Run linear regression analysis.
 *
 * Trains a linear regression model using the training data.
 * Tests the model using test data (if available) and predicts output labels for new data.
 */
const runLinearRegressionAnalysis = async (): Promise<void> => {
  // Load CSV file "cars.csv" containing training and test data
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

  // Create an instance of LinearRegression class and train the model
  const linearRegression: LinearRegression = new LinearRegression(
    tensor(features),
    tensor(labels),
    { learningRate: 0.1, iterations: 3, batchSize: 10 }
  );

  linearRegression.train();

  // Test the model using test data (if available)
  if (testFeatures && testLabels) {
    linearRegression.test(tensor(testFeatures), tensor(testLabels));
  }

  // Predict output labels for new data
  linearRegression.predict(tensor([[120, 2, 380]])).print();
};

/**
 * Run binary logistic regression analysis.
 *
 * Trains a binary logistic regression model using the training data.
 * Tests the model using test data (if available) and predicts output labels for new data.
 */
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

  // Create an instance of BinaryLogisticRegression class and train the model
  const binaryLogisticRegression: BinaryLogisticRegression =
    new BinaryLogisticRegression(tensor(features), tensor(labels), {
      learningRate: 0.5,
      iterations: 100,
      batchSize: 10,
      decisionBoundary: 0.6,
    });

  binaryLogisticRegression.train();

  // Test the model using test data (if available)
  if (testFeatures && testLabels) {
    binaryLogisticRegression.test(tensor(testFeatures), tensor(testLabels));
  }

  // Predict output labels for new data
  binaryLogisticRegression.predict(tensor([[88, 1.065, 127]])).print();
  binaryLogisticRegression.predict(tensor([[120, 2, 380]])).print();
};

/**
 * Run multinominal logistic regression analysis.
 *
 * Trains a multinominal logistic regression model using the training data.
 * Tests the model using test data (if available) and predicts output labels for new data.
 */
const runMultinominalLogisticRegressionAnalysis = async (): Promise<void> => {
  // Load CSV file "cars.csv" containing training and test data
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

  // Create an instance of MultinominalLogisticRegression class and train the model
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

  // Test the model using test data (if available)
  if (testFeatures && testLabels) {
    console.log(
      multinominalLogisticRegression.test(
        tensor(testFeatures),
        tensor(flatMap(testLabels))
      )
    );
  }

  // Predict output labels for new data
  multinominalLogisticRegression.predict(tensor([[150, 2.223, 200]])).print();
};

/**
 * Run image recognition analysis.
 *
 * Trains a multinominal logistic regression model for image recognition using the MNIST dataset.
 * Tests the model using test data and outputs the accuracy.
 */
const runImageRecognition = (): void => {
  // Load MNIST training data (5000 images)
  const mnistData: any = mnist.training(0, 5000);
  // Load MNIST test data (100 images)
  const testMnistData: any = mnist.testing(0, 100);

  // Extract features from training data
  const features: Array<Array<number>> = mnistData.images.values.map(
    (image: any) => flatMap(image)
  );
  // Extract labels from training data and convert them to one-hot encoded vectors
  const labels: Array<Array<number>> = mnistData.labels.values.map(
    (label: any) => {
      const row: Array<number> = new Array(10).fill(0);
      row[label] = 1;
      return row;
    }
  );

  // Extract features from test data
  const testFeatures: Array<Array<number>> = testMnistData.images.values.map(
    (image: any) => flatMap(image)
  );
  // Extract labels from test data and convert them to one-hot encoded vectors
  const testLabels: Array<Array<number>> = testMnistData.labels.values.map(
    (label: any) => {
      const row: Array<number> = new Array(10).fill(0);
      row[label] = 1;
      return row;
    }
  );

  // Create an instance of MultinominalLogisticRegression class and train the model
  const multinominalLogisticRegression: MultinominalLogisticRegression =
    new MultinominalLogisticRegression(tensor(features), tensor(labels), {
      learningRate: 1,
      iterations: 20,
      batchSize: 100,
    });

  multinominalLogisticRegression.train();

  // Test the model using test data and output the accuracy
  console.log(
    "Accuracy is: ",
    multinominalLogisticRegression.test(
      tensor(testFeatures),
      tensor(testLabels)
    )
  );
};

export {
  runKnnAnalysis,
  runLinearRegressionAnalysis,
  runBinaryLogisticRegressionAnalysis,
  runMultinominalLogisticRegressionAnalysis,
  runImageRecognition,
};
