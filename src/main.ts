import { analyzeDataset } from "./algorithm";

type Dataset = Array<Array<number>>;

let datasetToAnalyze: Dataset = [];

const fetchDataset = async (): Promise<void> => {
  await fetch("/dataset.json")
    .then((response) => response.json())
    .then((data) => {
      datasetToAnalyze = data.dataset;
    })
    .catch((error) => {
      console.error(error);
    });
};

const start = async (): Promise<void> => {
  await fetchDataset();
  analyzeDataset(datasetToAnalyze);
};

start();
