package org.minnen.pedanticnn;

import java.util.ArrayList;
import java.util.List;

public class Node
{
  public final List<Connection> parents = new ArrayList<Connection>();
  public final List<Connection> kids    = new ArrayList<Connection>();

  public final NeuralLayer      layer;
  public final int              index;
  public double                 weightedInput, activation, bias;

  private double                gradBias;

  public Node(NeuralLayer layer, int index)
  {
    this.layer = layer;
    this.index = index;
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

  public void resetForLearning()
  {
    gradBias = 0.0;
  }

  public boolean isOutputNode()
  {
    return layer.isOutputLayer;
  }

  /** Calculate activation from parents. */
  public void feedForward()
  {
    // Our activation is a linear function of parent activations.
    double z = bias;
    for (Connection c : parents) {
      assert c.kid == this;
      z += c.weight * c.parent.activation;
    }
    weightedInput = z;
    activation = sigmoid(z);
  }

  public void backprop(double expected)
  {
    double error;
    if (isOutputNode()) {
      error = costDeriv(expected) * sigmoidDeriv(weightedInput);
    } else {
      error = 0.0; // TODO
    }

    gradBias += error;
  }

  public void updateParams(double learningRate)
  {
    if (Math.abs(gradBias) > 1e-6) {
      //System.out.printf("Node %d.%d: gradBias=%f  Bias: %f -> %f\n",
      //    layer.index, index, gradBias, bias, bias - learningRate * gradBias);
      bias -= learningRate * gradBias;
    }    
  }

  public double cost(double expected)
  {
    // TODO support other cost functions.
    double diff = expected - activation;
    return 0.5 * diff * diff;
  }

  public double costDeriv(double expected)
  {
    // TODO support other cost functions.
    return activation - expected;
  }

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
