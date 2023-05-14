import { Tensor, Rank } from "@tensorflow/tfjs";

type Options = {
  learningRate: number;
};

class LinearRegression {
  private features: Tensor<Rank>;
  private labels: Tensor<Rank>;
  private options: Options;

  constructor(features: Tensor<Rank>, labels: Tensor<Rank>, options: Options) {
    this.features = features;
    this.labels = labels;
    this.options = options;
  }
}

export { LinearRegression };
