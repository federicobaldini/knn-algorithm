import {
  runLinearRegressionAnalysis,
  runBinaryLogisticRegressionAnalysis,
} from "./analyzer";

const start = (): void => {
  runLinearRegressionAnalysis();
  runBinaryLogisticRegressionAnalysis();
};

start();
