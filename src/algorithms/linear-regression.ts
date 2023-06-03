import { Tensor, Rank, ones, zeros } from "@tensorflow/tfjs";

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
  private features: Tensor<Rank>; // Input features for training data
  private labels: Tensor<Rank>; // Output labels for training data
  private options: Options; // Configuration options for the model
  private weights: Tensor<Rank>; // Contains the slope (m) and y-intercept (b) values.

  /**
   * Creates an instance of LinearRegression.
   *
   * @param features - The training data features.
   * @param labels - The training data labels.
   * @param options - The configuration options for the model.
   */
  constructor(features: Tensor<Rank>, labels: Tensor<Rank>, options?: Options) {
    this.features = this.processFeatures(features);
    this.labels = labels;

    // Set default options if not provided.
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );

    this.weights = zeros([2, 1]);
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
    const currentGuesses: Tensor<Rank> = this.features.matMul(this.weights);
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
    testFeatures = this.processFeatures(testFeatures);
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
  processFeatures(features: Tensor<Rank>): Tensor<Rank> {
    // Concatenate a column of ones to the features tensor
    features = features.concat(ones([features.shape[0], 1]), 1);
    return features;
  }
}

export { LinearRegression };
