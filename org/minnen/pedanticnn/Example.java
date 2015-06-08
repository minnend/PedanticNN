package org.minnen.pedanticnn;

public class Example
{
  public final int      label;
  public final double[] expected;
  public final double[] data;

  public Example(int label, double[] expected, double[] data)
  {
    this.label = label;
    this.expected = expected;
    this.data = data;    
  }

  public int numInputDims()
  {
    return data.length;
  }
  
  public int numOutputDims()
  {
    return expected.length;
  }
}
