package org.minnen.pedanticnn;

import java.util.ArrayList;
import java.util.List;

import org.minnen.pedanticnn.cost.CostFunction;

public class Node {
  public final List<Connection> parents = new ArrayList<Connection>();
  public final List<Connection> kids = new ArrayList<Connection>();

  public final NeuralLayer layer;
  public final int index;

  public double bias;
  public double weightedInput; // z = w*x+b
  public double activation; // sig(z)
  public double dz; // partial derivatice of cost wrt z

  public double gradBias; // grad for bias via backprop
  public double fdBias; // grad for bias via finite difference

  public Node(NeuralLayer layer, int index) {
    this.layer = layer;
    this.index = index;
  }

  public void connect(Node kid, double weightScale) {
    Connection c = new Connection(this, kid, weightScale);
    kids.add(c);
    kid.parents.add(c);
  }

  public void setActivation(double activation) {
    this.activation = activation;
  }

  public void resetForLearning() {
    gradBias = 0.0;
    fdBias = 0.0;
    for (Connection c : parents) {
      c.gradWeight = 0.0;
      c.fdWeight = 0.0;
    }
  }

  public boolean isInputNode() {
    return layer.isInputLayer;
  }

  public boolean isOutputNode() {
    return layer.isOutputLayer;
  }

  /** Calculate activation from parents. */
  public void feedForward() {
    // Activation is a linear function of parent activations.
    weightedInput = bias;
    for (Connection c : parents) {
      weightedInput += c.weight * c.parent.activation;
    }
    activation = MathUtils.sigmoid(weightedInput); // TODO support other activation functions
  }

  public void backprop(double expected) {
    if (isOutputNode()) {
      CostFunction cf = layer.network.costFunction;
      double da = cf.deriv(activation, expected);
      dz = da * MathUtils.sigmoidDeriv(weightedInput); // TODO support other activation functions
    } else {
      double sum = 0.0;
      for (Connection c : kids) {
        sum += c.weight * c.kid.dz;
      }
      dz = sum * MathUtils.sigmoidDeriv(weightedInput);
    }

    gradBias -= dz;
    for (Connection c : parents) {
      c.gradWeight -= c.parent.activation * dz;
    }
  }

  public void updateParams(double learningRate, double lambda, int trainSize, int batchSize) {
    bias -= learningRate / batchSize * gradBias;
    for (Connection c : parents) {
      c.weight =
          (1.0 - learningRate * lambda / trainSize) * c.weight
          - learningRate / batchSize * c.gradWeight;
    }
  }

  public String name() {
    return String.format("%d.%d", layer.index, index);
  }

  @Override
  public String toString() {
    return String.format(
        "[Node %s: b=%.3f  grad=%.5f (%.5f) (input=%.5f, a=%.6f)  Connections: %d/%d]", name(),
        bias, gradBias, fdBias, weightedInput, activation, parents.size(), kids.size());
  }
}
