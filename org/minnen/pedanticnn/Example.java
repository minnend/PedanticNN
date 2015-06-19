package org.minnen.pedanticnn;

public class Example {
  /** Human-identifiable class label (e.g. [0..9]) */
  public final int label;
  
  /** Expected activation in output layer (e.g. one-hot with 10 nodes) */
  public final double[] expected;
  
  /** Input feature vector (e.g. pixel intensity values) */
  public final double[] data;

  /**
   * Construct a new example from a label, expected activation values, and feature vector.
   * @param label human-identifiable class label (e.g. [0..9])
   * @param expected expected activation in output layer (e.g. one-hot with 10 nodes)
   * @param data input feature vector (e.g. pixel intensity values)
   */
  public Example(int label, double[] expected, double[] data) {
    this.label = label;
    this.expected = expected;
    this.data = data;
  }

  public int numInputDims() {
    return data.length;
  }

  public int numOutputDims() {
    return expected.length;
  }

  @Override
  public String toString() {
    if (data.length <= 6) {
      StringBuilder sb = new StringBuilder();
      for (int i=0; i<data.length; ++i) {
        sb.append(String.format("%.2f", data[i]));
        if (i < data.length - 1) {
          sb.append(",");
        }
      }
      return String.format("[Example: [%s] -> %dD  Label=%d]", sb.toString(), expected.length, label);      
    } else {
      return String.format("[Example: %dD -> %dD  Label=%d]", data.length, expected.length, label);
    }
  }
}
