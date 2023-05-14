import { Tensor, Rank } from "@tensorflow/tfjs";

type Options = {
  learningRate: number;
  iterations: number;
};

class LinearRegression {
  private features: Tensor<Rank>;
  private labels: Tensor<Rank>;
  private options: Options;

  constructor(features: Tensor<Rank>, labels: Tensor<Rank>, options?: Options) {
    this.features = features;
    this.labels = labels;
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );
  }

  gradientDescent(): void {}

  train(): void {
    for (let i = 0; i < this.options.iterations; i += 1) {
      this.gradientDescent();
    }
  }
}

export { LinearRegression };
