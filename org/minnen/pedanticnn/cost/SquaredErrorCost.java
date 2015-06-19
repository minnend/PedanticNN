package org.minnen.pedanticnn.cost;

public class SquaredErrorCost implements CostFunction {
  @Override
  public double f(double a, double y) {
    double diff = a - y;
    return 0.5 * diff * diff;
  }

  @Override
  public double deriv(double a, double y) {
    return a - y;
  }
}
