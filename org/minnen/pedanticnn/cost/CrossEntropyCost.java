package org.minnen.pedanticnn.cost;

public class CrossEntropyCost implements CostFunction
{

  @Override
  public double f(double x, double y)
  {
    double v = -y * Math.log(x) - (1.0 - y) * Math.log(1.0 - x);
    return Double.isNaN(v) ? 0.0 : v;
  }

  @Override
  public double deriv(double x, double y, double z)
  {
    return x - y;
  }

}
