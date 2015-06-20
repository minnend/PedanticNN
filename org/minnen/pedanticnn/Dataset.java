package org.minnen.pedanticnn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Dataset {
  public final List<Example> data = new ArrayList<Example>();
  public final int numInputDims;
  public final int numOutputDims;

  /**
   * Create a new dataset by copying a subset of an existing one.
   * @param dataset existing dataset that holds examples and labels
   * @param startIndex index of first example to copy
   * @param numExamples number of examples to copy
   */
  public Dataset(Dataset dataset, int startIndex, int numExamples) {
    for (int i = 0; i < numExamples; ++i) {
      data.add(dataset.get(startIndex + i));
    }
    numInputDims = dataset.numInputDims;
    numOutputDims = dataset.numOutputDims;
  }

  public Dataset(List<Integer> labels, List<double[]> inputs, int maxLabel) {
    if (labels.size() != inputs.size()) {
      throw new IllegalArgumentException(String.format(
          "Number of labels must match inputs (%d vs %d)", labels.size(), inputs.size()));
    }
    numInputDims = inputs.get(0).length;
    numOutputDims = maxLabel + 1;

    // Build expected output for each label.
    final double[][] expected = new double[numOutputDims][numOutputDims];
    for (int i = 0; i < numOutputDims; ++i) {
      expected[i][i] = 1.0;
    }

    for (int i = 0; i < labels.size(); ++i) {
      data.add(new Example(labels.get(i), expected[labels.get(i)], inputs.get(i)));
    }
  }

  public double[] mean() {
    double[] m = new double[numInputDims];
    for (Example example : data) {
      MathUtils.add(m, example.data);
    }
    return MathUtils.div(m, size());
  }

  public double[] variance() {
    double[] m = mean();
    double[] v = new double[numInputDims];
    for (Example example : data) {
      for (int i = 0; i < m.length; ++i) {
        double diff = example.data[i] - m[i];
        v[i] += diff * diff;
      }
    }
    return MathUtils.div(v, size() - 1);
  }

  /** Z-Normalize data = zero mean and unit variance. */
  public void znormalize() {
    zeroMean();
    unitVariance();
  }

  public void zeroMean() {
    double[] m = mean();
    for (Example example : data) {
      MathUtils.sub(example.data, m);
    }
  }

  public void unitVariance() {
    double[] m = mean();
    double[] stdev = MathUtils.sqrt(variance());
    for (Example example : data) {
      for (int i = 0; i < m.length; ++i) {
        if (Math.abs(stdev[i]) > 1e-9) {
          example.data[i] = (example.data[i] - m[i]) / stdev[i] + m[i];
        }
      }
    }
  }

  public int size() {
    return data.size();
  }

  public Example get(int i) {
    return data.get(i);
  }

  public void shuffleExamples() {
    Collections.shuffle(data);
  }
}
