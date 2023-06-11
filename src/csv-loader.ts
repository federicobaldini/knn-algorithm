import {
  dropRightWhile,
  first,
  isEqual,
  pullAt,
  isNaN,
  isNumber,
} from "lodash";
import fs from "fs";
import shuffleSeed from "shuffle-seed";

type Dataset = Array<Array<string | number>>;

/**
 * This function extracts specific columns from a CSV file and returns them as an array.
 * @param dataset The dataset to extract columns from.
 * @param columnNames An array of names of columns to extract.
 * @returns The extracted columns as an array.
 */
const extractColumns = (
  dataset: Dataset,
  columnNames: Array<string>
): Dataset => {
  // Get the headers of the dataset, which should be in the first row.
  const headers: Array<string> = first(dataset) as Array<string>;

  // Find the indexes of the columns to extract based on their names.
  const indexes: Array<number> = columnNames.map((columnName) =>
    headers.indexOf(columnName)
  );

  // Extract the specified columns from the dataset and return them.
  const extracted: Dataset = dataset.map((row) => pullAt(row, indexes));

  return extracted;
};

/**
 * This function loads a CSV file and returns its dataset in a structured format.
 * @param filename The name of the CSV file to load.
 * @param datasetColumns An array of names of columns containing the dataset to use for training/testing.
 * @param labelColumns An array of names of columns containing the labels for the dataset.
 * @param converters An object containing functions to use for converting specific columns to numbers.
 * @param shuffle A boolean indicating whether or not to shuffle the dataset.
 * @param splitTest An optional number indicating the size of the test set to split off from the dataset.
 * @returns An object containing the features and labels of the dataset, as well as optional test features/labels.
 */
const loadCSV = (
  filename: string,
  {
    datasetColumns = [],
    labelColumns = [],
    converters = {},
    shuffle = false,
    splitTest = undefined,
  }: {
    datasetColumns?: Array<string>;
    labelColumns?: Array<string>;
    converters?: { [key: string]: (...args: Array<any>) => any };
    shuffle?: boolean;
    splitTest?: number;
  }
): {
  features: Dataset;
  labels: Dataset;
  testFeatures?: Dataset;
  testLabels?: Dataset;
} => {
  // Read the CSV file and store its contents as a string.
  let stringDataset: string = fs.readFileSync(`./public/datasets/${filename}`, {
    encoding: "utf-8",
  });

  // Parse the CSV dataset into a 2D array of strings.
  let dataset: Array<Array<string>> = stringDataset
    .split("\n")
    .map((dataset: string): Array<string> => dataset.split(","));

  // Remove any rows that contain empty strings (which can happen if there are trailing commas in the CSV file).
  dataset = dropRightWhile(dataset, (value: Array<string>): boolean =>
    isEqual(value, [""])
  );

  // Get the headers of the dataset, which should be in the first row.
  const headers: Array<string> = first(dataset) as Array<string>;

  // Convert the dataset to an array of numbers, using any specified converters to convert specific columns.
  let convertedDataset: Dataset = dataset.map(
    (row: Array<string>, index: number): Array<string | number> => {
      if (index === 0) {
        // If this is the first row (i.e. the headers), leave it as strings.
        return row;
      }
      return row.map((element: string, index: number): string | number => {
        if (converters[headers[index]]) {
          // If there is a converter specified for this column, use it to convert the element to a number.
          const converted: string | number =
            converters[headers[index]](element);
          return isNaN(converted) ? element : converted;
        }

        // If there is no converter specified for this column, attempt to parse the element as a number.
        const result: number = parseFloat(element.replace('"', ""));
        return isNaN(result) ? element : result;
      });
    }
  );

  // Extract the label columns from the converted dataset and store them separately.
  let labels: Dataset = extractColumns(convertedDataset, labelColumns);
  // Extract the dataset columns from the converted dataset and store them separately.
  convertedDataset = extractColumns(convertedDataset, datasetColumns);

  // Remove the first row (i.e. the headers) from the converted dataset and labels.
  convertedDataset.shift();
  labels.shift();

  if (shuffle) {
    // Shuffle the converted dataset and labels using a specified seed.
    convertedDataset = shuffleSeed.shuffle(convertedDataset, "phrase");
    labels = shuffleSeed.shuffle(labels, "phrase");
  }

  if (splitTest !== undefined) {
    // Split the dataset and labels into a training set and test set.
    const trainSize = isNumber(splitTest)
      ? splitTest
      : Math.floor(convertedDataset.length / 2);

    return {
      features: convertedDataset.slice(trainSize),
      labels: labels.slice(trainSize),
      testFeatures: convertedDataset.slice(0, trainSize),
      testLabels: labels.slice(0, trainSize),
    };
  } else {
    // Return the full dataset and labels.
    return { features: convertedDataset, labels };
  }
};

export { loadCSV };
