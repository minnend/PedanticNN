package org.minnen.pedanticnn.cost;

public interface CostFunction
{
  /** @return f(x) with expected value y */
  public double f(double x, double y);
  
  /** @return f'(x) with expected value y and weighted input z */
  public double deriv(double x, double y, double z);
}
