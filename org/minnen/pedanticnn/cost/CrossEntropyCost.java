package org.minnen.pedanticnn.cost;

import org.minnen.pedanticnn.MathUtils;

public class CrossEntropyCost implements CostFunction {
  public final double almostZero = 1e-7;
  public final double almostOne = 1.0 - almostZero;

  @Override
  public double f(double a, double y) {
    double v = -y * Math.log(a) - (1.0 - y) * Math.log(1.0 - a);
    return Double.isNaN(v) ? 0.0 : v;
  }

  @Override
  public double deriv(double a, double y) {
    // Avoid divide-by-zero by clamping a to (0..1).
    a = MathUtils.clip(a, almostZero, almostOne);
    return y / a - (1.0 - y) / (1.0 - a);
  }
}
