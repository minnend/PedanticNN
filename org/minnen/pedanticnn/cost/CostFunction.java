package org.minnen.pedanticnn.cost;

public interface CostFunction {
  /** @return f(x) with activation a and expected value y. */
  public double f(double a, double y);

  /** @return f'(x) with activation a and expected value y. */
  public double deriv(double a, double y);
}
