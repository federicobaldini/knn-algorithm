import { tensor } from "@tensorflow/tfjs";
import { loadCSV } from "./csv-loader";
import { knn } from "./algorithm";

type Dataset = Array<Array<string | number>>;

const start = async (): Promise<void> => {
  let {
    features,
    labels,
    testFeatures,
    testLabels,
  }: {
    features: Dataset;
    labels: Dataset;
    testFeatures?: Dataset;
    testLabels?: Dataset;
  } = loadCSV("kc_house_data.csv", {
    datasetColumns: ["lat", "long", "sqft_lot", "sqft_living"],
    labelColumns: ["price"],
    shuffle: true,
    splitTest: 10,
  });

  if (testFeatures !== undefined) {
    testFeatures.forEach(
      (testFeature: Array<string | number>, index: number): void => {
        const result: number = knn(
          tensor(features),
          tensor(labels),
          tensor(testFeature),
          10
        );
        if (testLabels !== undefined) {
          const error: number =
            ((testLabels[index][0] as number) - result) /
            (testLabels[index][0] as number);
          if (Math.abs(error * 100) < 10) {
            console.log(
              `Success - Error for feature ${index}: ${(error * 100).toFixed(
                2
              )}%`
            );
          } else {
            console.log(
              `Error for feature ${index}: ${(error * 100).toFixed(2)}%`
            );
          }
        }
      }
    );
  }
};

start();
