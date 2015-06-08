package org.minnen.pedanticnn.cost;

import org.minnen.pedanticnn.MathUtils;

public class SquaredErrorCost implements CostFunction
{

  @Override
  public double f(double x, double y)
  {
    double diff = x - y;
    return 0.5 * diff * diff;
  }

  @Override
  public double deriv(double x, double y, double z)
  {
    return (x - y) * MathUtils.sigmoidDeriv(z);
  }

}
