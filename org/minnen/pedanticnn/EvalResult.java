package org.minnen.pedanticnn;

public class EvalResult
{
  public final double cost;
  public final int    numCorrect;
  public final int    numExamples;

  public EvalResult(double cost, int numCorrect, int numExamples)
  {
    this.cost = cost;
    this.numCorrect = numCorrect;
    this.numExamples = numExamples;
  }

  public double accuracy()
  {
    if (numExamples < 1) {
      return 0.0;
    } else {
      return 100.0 * numCorrect / numExamples;
    }

  }
}
