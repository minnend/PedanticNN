package org.minnen.pedanticnn;

import java.util.ArrayList;
import java.util.List;

import org.minnen.pedanticnn.cost.CostFunction;

public class Node
{
  public final List<Connection> parents = new ArrayList<Connection>();
  public final List<Connection> kids    = new ArrayList<Connection>();

  public final NeuralLayer      layer;
  public final int              index;
  public double                 weightedInput, activation, bias, error;

  public double                 gradBias;                              // grad for bias via backprop
  public double                 fdBias;                                // grad for bias via finite difference

  public Node(NeuralLayer layer, int index)
  {
    this.layer = layer;
    this.index = index;
  }

  public void connect(Node kid, double weightScale)
  {
    Connection c = new Connection(this, kid, weightScale);
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
    fdBias = 0.0;
    for (Connection c : parents) {
      c.gradWeight = 0.0;
      c.fdWeight = 0.0;
    }
  }

  public boolean isInputNode()
  {
    return layer.isInputLayer;
  }

  public boolean isOutputNode()
  {
    return layer.isOutputLayer;
  }

  /** Calculate activation from parents. */
  public void feedForward()
  {
    // Activation is a linear function of parent activations.
    weightedInput = bias;
    for (Connection c : parents) {
      weightedInput += c.weight * c.parent.activation;
    }
    activation = MathUtils.sigmoid(weightedInput);  // TODO support other activation functions
  }

  public void backprop(double expected)
  {
    if (isOutputNode()) {
      CostFunction cf = layer.network.costFunction;
      error = cf.deriv(activation, expected, weightedInput);
    } else {
      double sum = 0.0;
      for (Connection c : kids) {
        sum += c.weight * c.kid.error;
      }
      error = sum * MathUtils.sigmoidDeriv(activation);
    }

    gradBias += error;
    for (Connection c : parents) {
      c.gradWeight += c.parent.activation * error;
    }
  }

  public void updateParams(double learningRate, double lambda, int trainSize, int batchSize)
  {
    bias -= learningRate / batchSize * gradBias;
    for (Connection c : parents) {
      c.weight = (1.0 - learningRate * lambda / trainSize) * c.weight - learningRate / batchSize * c.gradWeight;
    }
  }
}
