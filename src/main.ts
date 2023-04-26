import { analyzeDataset } from "./algorithm";

let dataToAnalyze: Array<Array<number>> = [];

const fetchDataset = async (): Promise<void> => {
  await fetch("/dataset.json")
    .then((response) => response.json())
    .then((data) => {
      dataToAnalyze = data.dataset;
    })
    .catch((error) => {
      console.error(error);
    });
};

const start = async (): Promise<void> => {
  await fetchDataset();
  const predictedLabel: number = analyzeDataset(dataToAnalyze);
  console.log(predictedLabel);
};

start();
