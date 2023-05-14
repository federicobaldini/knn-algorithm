import { Tensor, Rank } from "@tensorflow/tfjs";

type Options = {
  learningRate: number;
  iterations: number;
};

type Dataset = Array<Array<string | number>>;

class LinearRegression {
  private features: Dataset;
  private labels: Dataset;
  private options: Options;
  private m: number;
  private b: number;

  constructor(features: Dataset, labels: Dataset, options?: Options) {
    this.features = features;
    this.labels = labels;
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );
    this.m = 0;
    this.b = 0;
  }

  gradientDescent(): void {
    // TODO: generalize it
    const currentGuessesForMPG = this.features.map(
      (row: Array<string | number>): number => {
        if (Number(row[0])) {
          return this.m * (row[0] as number) + this.b;
        }
        return 0;
      }
    );
  }

  train(): void {
    for (let i = 0; i < this.options.iterations; i += 1) {
      this.gradientDescent();
    }
  }
}

export { LinearRegression };
