import { Tensor, Rank, ones, zeros, moments } from "@tensorflow/tfjs";

// Define the Options type for configuration.
type Options = {
  learningRate: number;
  iterations: number;
  batchSize?: number;
  decisionBoundary?: number;
};

/**
 * Represents a multinominal logistic regression model.
 *
 * The model learns the relationship between independent variables (features)
 * and dependent variables (labels) using gradient descent optimization.
 */
class MultinominalLogisticRegression {
  private features: Tensor<Rank>; // Input features for training data.
  private labels: Tensor<Rank>; // Output labels for training data.
  private options: Options; // Configuration options for the model.
  private weights: Tensor<Rank>; // Contains the slope (m) and y-intercept (b) values.
  private mean: Tensor<Rank> | undefined; // Mean tensor for standardization.
  private variance: Tensor<Rank> | undefined; // Variance tensor for standardization.
  private costHistory: Array<number>; // Stores the history of cross entropy during training.

  /**
   * Creates an instance of LogisticRegression.
   *
   * @param features - The training data features.
   * @param labels - The training data labels.
   * @param options - The configuration options for the model.
   */
  constructor(features: Tensor<Rank>, labels: Tensor<Rank>, options?: Options) {
    this.features = this.initFeatures(features);
    this.labels = labels;
    this.costHistory = [];

    // Set default options if not provided
    this.options = Object.assign(
      {
        learningRate: 10,
        iterations: 100,
        batchSize: this.features.shape[0],
        decisionBoundary: 0.5,
      },
      options
    );

    if ((this.options.batchSize as number) < 1) {
      throw new Error("The batchSize must be greater than zero.");
    }

    if ((this.options.batchSize as number) > this.features.shape[0]) {
      this.options.batchSize = this.features.shape[0];
    }

    // Initialize the weights tensor based on the number of features
    if (this.features.shape[1] && this.labels.shape[1]) {
      this.weights = zeros([this.features.shape[1], this.labels.shape[1]]);
    } else {
      throw new Error(
        "The 'features' or 'labels' tensor has an undefined shape[1]."
      );
    }

    this.weights.print();
  }

  /**
   * Performs gradient descent optimization to update the model parameters.
   */
  gradientDescent(features: Tensor<Rank>, labels: Tensor<Rank>): void {
    // Calculate the current predictions
    const currentGuesses: Tensor<Rank> = features
      .matMul(this.weights)
      .sigmoid();

    // Calculate the differences between predictions and labels
    const differences: Tensor<Rank> = currentGuesses.sub(labels);

    // Calculate the slope (gradients) of MSE using the differences and features
    const slope: Tensor<Rank> = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    // Update the weights using the calculated slope and learning rate
    this.weights = this.weights.sub(slope.mul(this.options.learningRate));
  }

  /**
   * Trains the linear regression model by performing gradient descent optimization.
   */
  train(): void {
    const batchSize: number = this.options.batchSize as number;

    // Calculate the number of batches based on the total number of training data and batch size
    const batchQuantity: number = Math.floor(
      this.features.shape[0] / batchSize
    );

    // Iterate over the specified number of iterations
    for (let i = 0; i < this.options.iterations; i += 1) {
      // Iterate over each batch.
      for (let j = 0; j < batchQuantity; j += 1) {
        // Calculate the starting index of the current batch
        const startIndex: number = j * batchSize;
        // Create a slice of features tensor corresponding to the current batch
        const featuresSlice: Tensor<Rank> = this.features.slice(
          [startIndex, 0],
          [batchSize, -1]
        );

        // Create a slice of labels tensor corresponding to the current batch
        const labelsSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

        // Perform gradient descent optimization using the current batch
        this.gradientDescent(featuresSlice, labelsSlice);
      }
      // Record the cross entropy during training
      this.recordCost();

      // Update the learning rate based on the MSE history
      this.updateLearningRate();
    }
  }

  /**
   * Predicts the output labels for the given input observations.
   *
   * @param observations - The input observations for prediction.
   * @returns The predicted output labels.
   */
  predict(observations: Tensor<Rank>): Tensor<Rank> {
    // Initialize the features tensor for the input observations
    const features: Tensor<Rank> = this.initFeatures(observations);

    // Perform matrix multiplication of the features tensor and weights tensor
    // to generate the predicted output labels
    const predictions: Tensor<Rank> = features
      .matMul(this.weights)
      .sigmoid()
      .greater(this.options.decisionBoundary as number)
      .cast("float32");

    return predictions;
  }

  /**
   * Tests the trained linear regression model using test features and labels.
   *
   * @param testFeatures - The test data features.
   * @param testLabels - The test data labels.
   */
  test(testFeatures: Tensor<Rank>, testLabels: Tensor<Rank>): number {
    const predictions: Tensor<Rank> = this.predict(testFeatures);
    const incorrect: number = predictions
      .sub(testLabels)
      .abs()
      .sum()
      .arraySync() as number;

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  /**
   * Adds a column of ones to the features tensor for linear regression.
   *
   * @param features - The input features tensor.
   * @returns The modified features tensor with an additional column of ones.
   */
  initFeatures(features: Tensor<Rank>): Tensor<Rank> {
    // Standardize the features tensor
    features = this.standardize(features);

    // Concatenate a column of ones to the features tensor
    features = features.concat(ones([features.shape[0], 1]), 1);

    return features;
  }

  /**
   * Initializes the mean and variance tensors for feature standardization.
   *
   * @param features - The input features tensor.
   */
  initStandardizationParameters(features: Tensor<Rank>): void {
    // Calculate the mean and variance tensors
    const { mean, variance }: { mean: Tensor<Rank>; variance: Tensor<Rank> } =
      moments(features, 0);

    // Store the mean and variance tensors
    this.mean = mean;
    this.variance = variance;
  }

  /**
   * Standardizes the features tensor by subtracting the mean and dividing by the standard deviation.
   *
   * @param features - The input features tensor.
   * @returns The standardized features tensor.
   */
  standardize(features: Tensor<Rank>): Tensor<Rank> {
    // Check if the mean and variance tensors are not initialized
    if (!this.mean || !this.variance) {
      // Calculate and store the mean and variance tensors
      this.initStandardizationParameters(features);
    }

    // Check if the mean and variance tensors are available
    if (this.mean && this.variance) {
      // Perform standardization on the features tensor
      // by subtracting the mean and dividing by the standard deviation
      return features.sub(this.mean).div(this.variance.pow(0.5));
    }

    // If mean and variance tensors are not available, return the original features tensor
    return features;
  }

  /**
   * Records the value of the cross entropy cost during the training process.
   * The cost is calculated based on the current predictions and labels.
   * The calculated cost value is added to the beginning of the costHistory array.
   */
  recordCost(): void {
    // Calculate the current predictions (probabilities) using sigmoid activation
    const guesses: Tensor<Rank> = this.features.matMul(this.weights).sigmoid();

    // Calculate the termOne of the cross entropy cost
    const termOne = this.labels.transpose().matMul(guesses.log());

    // Calculate the termTwo of the cross entropy cost
    const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(guesses.mul(-1).add(1).log());

    // Calculate the overall cross entropy cost
    const cost: number = (
      termOne
        .add(termTwo)
        .div(this.features.shape[0])
        .mul(-1)
        .arraySync() as Array<number>
    )[0];

    // Add the cost value to the beginning of the costHistory array
    this.costHistory.unshift(cost);
  }

  /**
   * Updates the learning rate based on the mean squared error history.
   * If the last MSE is greater than the second last MSE, the learning rate is halved.
   * If the last MSE is not greater than the second last MSE, the learning rate is increased by 5%.
   */
  updateLearningRate() {
    // Check if enough MSE values are available for comparison
    if (this.costHistory.length < 2) {
      return;
    }

    // Compare the last MSE value with the second last MSE value
    if (this.costHistory[0] > this.costHistory[1]) {
      // Last MSE is greater than the second last MSE, indicating increasing MSE
      this.options.learningRate /= 2; // Halve the learning rate
    } else {
      // Last MSE is not greater than the second last MSE, indicating decreasing or stable MSE
      this.options.learningRate *= 1.05; // Increase the learning rate by 5%
    }
  }
}

export { MultinominalLogisticRegression };
