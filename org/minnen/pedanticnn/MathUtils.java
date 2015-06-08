package org.minnen.pedanticnn;

public class MathUtils
{
  public static double sigmoid(double x)
  {
    return 1.0 / (1.0 + Math.exp(-x));
  }

  public static double sigmoidDeriv(double x)
  {
    double s = sigmoid(x);
    return s * (1.0 - s);
  }
}
