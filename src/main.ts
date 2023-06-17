import {
  runLinearRegressionAnalysis,
  runLogisticRegressionAnalysis,
} from "./analyzer";

const start = (): void => {
  runLinearRegressionAnalysis();
  runLogisticRegressionAnalysis();
};

start();
