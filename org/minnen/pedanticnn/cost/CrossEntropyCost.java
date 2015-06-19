package org.minnen.pedanticnn.cost;

public class CrossEntropyCost implements CostFunction
{

  @Override
  public double f(double a, double y)
  {
    double v = -y * Math.log(a) - (1.0 - y) * Math.log(1.0 - a);
    return Double.isNaN(v) ? 0.0 : v;
  }

  @Override
  public double deriv(double a, double y, double z)
  {
    return a - y;
  }

}
