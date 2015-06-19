package org.minnen.pedanticnn;

import java.io.BufferedReader;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.minnen.pedanticnn.cost.CostFunction;
import org.minnen.pedanticnn.cost.CrossEntropyCost;

public class LearnNN {
  private static Dataset LoadData(String filename) throws IOException {
    try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
      reader.readLine(); // skip header

      List<Integer> labels = new ArrayList<Integer>();
      List<double[]> inputs = new ArrayList<double[]>();
      int maxLabel = -1;

      String line;
      while ((line = reader.readLine()) != null) {
        String[] fields = line.split(",");

        // Load label.
        int label = Integer.parseInt(fields[0]);
        labels.add(label);
        if (label > maxLabel) {
          maxLabel = label;
        }

        // Load feature vector.
        double[] data = new double[fields.length - 1];
        for (int i = 0; i < data.length; ++i) {
          int v = Integer.parseInt(fields[i + 1]);
          data[i] = v / 255.0;
        }
        inputs.add(data);
      }

      return new Dataset(labels, inputs, maxLabel);
    }
  }

  private static Dataset LoadToyData() {
    List<Integer> labels = new ArrayList<>();
    List<double[]> inputs = new ArrayList<>();

    inputs.add(new double[] {3, 2});
    inputs.add(new double[] {3, 0});
    inputs.add(new double[] {0, 2});
    inputs.add(new double[] {-1, 0});
    inputs.add(new double[] {-1, -2});

    labels.add(1);
    labels.add(1);
    labels.add(0);
    labels.add(0);
    labels.add(0);

    return new Dataset(labels, inputs, 1);
  }

  public static EvalResult evaluate(Prediction[] preds, Dataset dataset) {
    double cost = 0.0;
    int nc = 0;
    final int N = dataset.size();
    for (int i = 0; i < N; ++i) {
      // System.err.printf("%d: %d vs %d\n", i, preds[i].label, dataset.get(i).label);
      if (dataset.get(i).label == preds[i].label) {
        ++nc;
      }
      cost += preds[i].cost;
    }
    return new EvalResult(cost, nc, N);
  }

  public static void main(String[] args) throws IOException {
    Dataset data = LoadData(args[0]);
    data.shuffleExamples();
    int numTrain = (int) (data.size() * 0.8);
    int numTest = data.size() - numTrain;
    Dataset dataTrain = new Dataset(data, 0, numTrain);
    Dataset dataTest = new Dataset(data, numTrain, numTest);
    System.out.printf("Training examples: %d @ %dD -> %dD\n", dataTrain.size(),
        dataTrain.numInputDims, dataTrain.numOutputDims);
    System.out.printf("Test examples: %d @ %dD -> %dD\n", dataTest.size(), dataTest.numInputDims,
        dataTest.numOutputDims);

    CostFunction costFunc = new CrossEntropyCost();
    NeuralNetwork network =
        new NeuralNetwork(new int[] {dataTrain.numInputDims, dataTrain.numOutputDims}, costFunc);
    System.out.println(network);

    double learningRate = 0.1;
    double lambda = 0.0;
    int batchSize = 100;
    boolean checkGradients = false;
    int numEpochs = 1000;
    network.train(dataTrain, dataTest, learningRate, lambda, batchSize, checkGradients, numEpochs);
  }
}
