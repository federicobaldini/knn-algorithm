import { sum } from "lodash";

// Define the Options type for configuration.
type Options = {
  learningRate: number;
  iterations: number;
};

// Define the Dataset type for input data.
type Dataset = Array<Array<string | number>>;

/**
 * Represents a linear regression model.
 *
 * The model learns the relationship between independent variables (features)
 * and dependent variables (labels) using gradient descent optimization.
 */
class LinearRegression {
  private features: Dataset;
  private labels: Dataset;
  private options: Options;
  private m: number;
  private b: number;

  /**
   * Creates an instance of LinearRegression.
   *
   * @param features - The training data features.
   * @param labels - The training data labels.
   * @param options - The configuration options for the model.
   */
  constructor(features: Dataset, labels: Dataset, options?: Options) {
    this.features = features;
    this.labels = labels;
    // Set default options if not provided.
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );
    // Set initial slope (m) and y-intercept (b) values.
    this.m = 0;
    this.b = 0;
  }

  getM(): number {
    return this.m;
  }

  getB(): number {
    return this.b;
  }

  /**
   * Performs gradient descent optimization to update the model parameters.
   */
  gradientDescent(): void {
    // TODO: generalize it

    // Calculate the current guesses for the dependent variable.
    const currentGuessesForMPG = this.features.map(
      (row: Array<string | number>): number => {
        if (Number(row[0])) {
          // Calculate the predicted value (guess) for a row.
          return this.m * (row[0] as number) + this.b;
        }
        // Return 0 for non-numeric values.
        return 0;
      }
    );

    // Calculate the slope of the Mean Squared Error (MSE) with respect to y-intercept (b).
    const mseSlopeByB: number =
      (sum(
        currentGuessesForMPG.map((guess: number, index: number): number => {
          if (Number(this.labels[index][0])) {
            // Calculate the partial derivative of the error with respect to y-intercept (b).
            return guess - (this.labels[index][0] as number);
          }
          // Return 0 for non-numeric values.
          return 0;
        })
      ) *
        2) /
      this.features.length;

    // Calculate the slope of the Mean Squared Error (MSE) with respect to slope (m).
    const mseSlopeByM: number =
      (sum(
        currentGuessesForMPG.map((guess: number, index: number): number => {
          if (Number(this.features[index][0])) {
            // Calculate the partial derivative of the error with respect to slope (m).
            return (
              -1 *
              (this.features[index][0] as number) *
              ((this.labels[index][0] as number) - guess)
            );
          }
          // Return 0 for non-numeric values.
          return 0;
        })
      ) *
        2) /
      this.features.length;

    // Update the slope (m) and y-intercept (b) values using gradient descent.
    this.m = this.m - mseSlopeByM * this.options.learningRate;
    this.b = this.b - mseSlopeByB * this.options.learningRate;
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
}

export { LinearRegression };
