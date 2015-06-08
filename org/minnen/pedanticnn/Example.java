package org.minnen.pedanticnn;

public class Example
{
  public final int      label;
  public final double[] data;

  public Example(int label, int numDims)
  {
    this.label = label;
    data = new double[numDims];
  }

  public Example(int label, double[] data)
  {
    this.label = label;
    this.data = data;
  }

  public int numDims()
  {
    return data.length;
  }
}
