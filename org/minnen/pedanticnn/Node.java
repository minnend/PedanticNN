package org.minnen.pedanticnn;

import java.util.ArrayList;
import java.util.List;

public class Node
{
  public final List<Connection> parents = new ArrayList<Connection>();
  public final List<Connection> kids    = new ArrayList<Connection>();

  public final NeuralLayer      layer;
  public final int              index;
  public double                 activation, bias;

  public Node(NeuralLayer layer, int index)
  {
    this.layer = layer;
    this.index = index;
    activation = 0.0;
    bias = 0.0;
  }

  public void connect(Node kid)
  {
    Connection c = new Connection(this, kid);
    kids.add(c);
    kid.parents.add(c);
  }

  public void setActivation(double activation)
  {
    this.activation = activation;
  }

  /** Calculate activation from parents. */
  public double feedForward()
  {
    double sum = 0.0;
    for (Connection c : parents) {
      assert c.kid == this;
      sum += c.weight * c.parent.activation;
    }
    activation = sigmoid(sum + bias);
    return activation;
  }

  public static double sigmoid(double x)
  {
    return 1.0 / (1.0 + Math.exp(-x));
  }
}
