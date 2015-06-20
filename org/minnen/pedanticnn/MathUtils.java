package org.minnen.pedanticnn;

public class MathUtils {
  public static double sigmoid(double x) {
    return 1.0 / (1.0 + Math.exp(-x));
  }

  public static double sigmoidDeriv(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
  }

  public static double clip(double x, double vmin, double vmax) {
    return Math.min(vmax, Math.max(vmin, x));
  }

  public static double[] add(double[] x, double[] dx) {
    assert x.length == dx.length;
    for (int i = 0; i < x.length; ++i) {
      x[i] += dx[i];
    }
    return x;
  }

  public static double[] sub(double[] x, double[] dx) {
    assert x.length == dx.length;
    for (int i = 0; i < x.length; ++i) {
      x[i] -= dx[i];
    }
    return x;
  }

  public static double[] mul(double[] x, double[] m) {
    assert x.length == m.length;
    for (int i = 0; i < x.length; ++i) {
      x[i] *= m[i];
    }
    return x;
  }

  public static double[] div(double[] x, double[] m) {
    assert x.length == m.length;
    for (int i = 0; i < x.length; ++i) {
      if (Math.abs(m[i]) > 1e-9) {
        x[i] /= m[i];
      }
    }
    return x;
  }

  public static double[] add(double[] x, double dx) {
    for (int i = 0; i < x.length; ++i) {
      x[i] += dx;
    }
    return x;
  }

  public static double[] sub(double[] x, double dx) {
    for (int i = 0; i < x.length; ++i) {
      x[i] -= dx;
    }
    return x;
  }

  public static double[] mul(double[] x, double m) {
    for (int i = 0; i < x.length; ++i) {
      x[i] *= m;
    }
    return x;
  }

  public static double[] div(double[] x, double m) {
    for (int i = 0; i < x.length; ++i) {
      x[i] /= m;
    }
    return x;
  }

  public static double[] sqrt(double[] x) {
    for (int i = 0; i < x.length; ++i) {
      x[i] = Math.sqrt(x[i]);
    }
    return x;
  }

  public static double l2norm(double[] x) {
    double sum = 0.0;
    for (int i = 0; i < x.length; ++i) {
      sum += x[i] * x[i];
    }
    return Math.sqrt(sum);
  }

  public static double l1norm(double[] x) {
    double sum = 0.0;
    for (int i = 0; i < x.length; ++i) {
      sum += Math.abs(x[i]);
    }
    return sum;
  }
}
