package org.minnen.pedanticnn;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.minnen.pedanticnn.cost.CrossEntropyCost;
import org.minnen.pedanticnn.cost.SquaredErrorCost;

public class LearnNN
{
  private static Dataset LoadData(String filename) throws IOException
  {
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

  public static void evaluate(Prediction[] preds, Dataset dataset)
  {
    double cost = 0.0;
    int nc = 0;
    final int N = dataset.size();
    for (int i = 0; i < N; ++i) {
      // System.err.printf("%d: %d vs %d\n", i, preds[i], examples.get(i).label);
      if (dataset.get(i).label == preds[i].label) {
        ++nc;
      }
      cost += preds[i].cost;
    }
    System.out.printf("Cost=%f   Accuracy: %d / %d = %.2f%%\n", cost, nc, N, 100.0 * nc / N);
  }

  public static void main(String[] args) throws IOException
  {
    String trainFile = args[0];
    Dataset dataset = LoadData(trainFile);
    System.err.printf("Training example: %d @ %dD -> %dD\n", dataset.size(), dataset.numInputDims,
        dataset.numOutputDims);

    NeuralNetwork network = new NeuralNetwork(
        new int[] { dataset.numInputDims, dataset.numOutputDims },
        //new SquaredErrorCost());
        new CrossEntropyCost());
    double learningRate = 0.01;
    int batchSize = 200;
    int numEpochs = 1000;
    network.train(dataset, learningRate, batchSize, numEpochs);
    evaluate(network.predict(dataset), dataset);
  }
}
