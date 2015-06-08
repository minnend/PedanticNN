package org.minnen.pedanticnn;

public class Prediction
{
  public final int      label;
  public final double[] output;
  public final double   cost;

  public Prediction(int label, double[] output, double cost)
  {
    this.label = label;
    this.output = output.clone();
    this.cost = cost;
  }
}
