import { Tensor, Rank, ones, zeros, moments } from "@tensorflow/tfjs";

// Define the Options type for configuration.
type Options = {
  learningRate: number;
  iterations: number;
};

/**
 * Represents a linear regression model.
 *
 * The model learns the relationship between independent variables (features)
 * and dependent variables (labels) using gradient descent optimization.
 */
class LinearRegression {
  private features: Tensor<Rank>; // Input features for training data.
  private labels: Tensor<Rank>; // Output labels for training data.
  private options: Options; // Configuration options for the model.
  private weights: Tensor<Rank>; // Contains the slope (m) and y-intercept (b) values.
  private mean: Tensor<Rank> | undefined; // Mean tensor for standardization.
  private variance: Tensor<Rank> | undefined; // Variance tensor for standardization.
  private mseHistory: Array<number>; // Stores the history of mean squared errors during training.

  /**
   * Creates an instance of LinearRegression.
   *
   * @param features - The training data features.
   * @param labels - The training data labels.
   * @param options - The configuration options for the model.
   */
  constructor(features: Tensor<Rank>, labels: Tensor<Rank>, options?: Options) {
    this.features = this.initFeatures(features);
    this.labels = labels;
    this.mseHistory = [];

    // Set default options if not provided.
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );

    // Initialize the weights tensor based on the number of features.
    if (this.features.shape[1]) {
      this.weights = zeros([this.features.shape[1], 1]);
    } else {
      throw new Error("The 'features' tensor has an undefined shape[1].");
    }

    this.mean = undefined;
    this.variance = undefined;
  }

  /**
   * Gets the slope (m) of the linear regression line.
   */
  getM(): number {
    return (this.weights.arraySync() as Array<number>)[0];
  }

  /**
   * Gets the y-intercept (b) of the linear regression line.
   */
  getB(): number {
    return (this.weights.arraySync() as Array<number>)[1];
  }

  /**
   * Performs gradient descent optimization to update the model parameters.
   */
  gradientDescent(): void {
    // Calculate the current predictions.
    const currentGuesses: Tensor<Rank> = this.features.matMul(this.weights);

    // Calculate the differences between predictions and labels.
    const differences: Tensor<Rank> = currentGuesses.sub(this.labels);

    // Calculate the slopes (gradients) using the differences and features.
    const slopes: Tensor<Rank> = this.features
      .transpose()
      .matMul(differences)
      .div(this.features.shape[0]);

    // Update the weights using the calculated slopes and learning rate.
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  /**
   * Trains the linear regression model by performing gradient descent optimization.
   */
  train(): void {
    for (let i = 0; i < this.options.iterations; i += 1) {
      // Perform gradient descent for the specified number of iterations.
      this.gradientDescent();
      this.recordMSE();
      this.updateLearningRate();
    }
  }

  /**
   * Tests the trained linear regression model using test features and labels.
   *
   * @param testFeatures - The test data features.
   * @param testLabels - The test data labels.
   * @returns The coefficient of determination (R^2) for the predictions.
   */
  test(testFeatures: Tensor<Rank>, testLabels: Tensor<Rank>): number {
    testFeatures = this.initFeatures(testFeatures);
    const predictions: Tensor<Rank> = testFeatures.matMul(this.weights);

    // Calculate the sum of squares of residuals (label - predicted)^2
    const sumOfSquaresOfResiduals: number = testLabels
      .sub(predictions)
      .pow(2)
      .sum()
      .arraySync() as number;

    // Calculate the total sum of squares (label - average)^2
    const totalSumOfSquares: number = testLabels
      .sub(testLabels.mean())
      .pow(2)
      .sum()
      .arraySync() as number;

    // Calculate the coefficient of determination (R^2)
    return 1 - sumOfSquaresOfResiduals / totalSumOfSquares;
  }

  /**
   * Adds a column of ones to the features tensor for linear regression.
   *
   * @param features - The input features tensor.
   * @returns The modified features tensor with an additional column of ones.
   */
  initFeatures(features: Tensor<Rank>): Tensor<Rank> {
    // Standardize the features tensor.
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
    // Calculate the mean and variance tensors.
    const { mean, variance }: { mean: Tensor<Rank>; variance: Tensor<Rank> } =
      moments(features, 0);

    // Store the mean and variance tensors.
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
   * Records the mean squared error (MSE) during training.
   */
  recordMSE(): void {
    // Calculate the mean squared error (MSE) using the current weights and training data.
    const mse: number = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .arraySync() as number;

    // Store the MSE in the history array.
    this.mseHistory.unshift(mse);
  }

  /**
   * Updates the learning rate based on the mean squared error history.
   * If the last MSE is greater than the second last MSE, the learning rate is halved.
   * If the last MSE is not greater than the second last MSE, the learning rate is increased by 5%.
   */
  updateLearningRate() {
    // Check if enough MSE values are available for comparison
    if (this.mseHistory.length < 2) {
      return;
    }

    // Compare the last MSE value with the second last MSE value
    if (this.mseHistory[0] > this.mseHistory[1]) {
      // Last MSE is greater than the second last MSE, indicating increasing MSE
      this.options.learningRate /= 2; // Halve the learning rate
    } else {
      // Last MSE is not greater than the second last MSE, indicating decreasing or stable MSE
      this.options.learningRate *= 1.05; // Increase the learning rate by 5%
    }
  }
}

export { LinearRegression };
