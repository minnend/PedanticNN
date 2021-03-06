package org.minnen.pedanticnn;

public class NeuralLayer
{
  private final Node[]       nodes;
  public final int           index;
  public final boolean       isInputLayer;
  public final boolean       isOutputLayer;
  public final NeuralNetwork network;

  public NeuralLayer(int index, int numNodes, boolean isInputLayer, boolean isOutputLayer, NeuralNetwork network)
  {
    this.index = index;
    this.isInputLayer = isInputLayer;
    this.isOutputLayer = isOutputLayer;
    nodes = new Node[numNodes];
    for (int i = 0; i < numNodes; ++i) {
      nodes[i] = new Node(this, i);
    }
    this.network = network;
  }

  public Node node(int i)
  {
    return nodes[i];
  }

  public int size()
  {
    return nodes.length;
  }

  public void setActivations(double[] data)
  {
    if (data.length != nodes.length)
      throw new IllegalArgumentException(String.format("%dD input for %d nodes", data.length, nodes.length));

    for (int i = 0; i < data.length; ++i) {
      nodes[i].setActivation(data[i]);
    }
  }

  public double[] getActivations()
  {
    double[] activations = new double[nodes.length];
    for (int i = 0; i < nodes.length; ++i) {
      activations[i] = nodes[i].activation;
    }
    return activations;
  }

  public void resetForLearning()
  {
    for (Node node : nodes) {
      node.resetForLearning();
    }
  }

  /** Calculate activations for nodes in this layer from previous layer. */
  public void feedForward()
  {
    for (Node node : nodes) {
      node.feedForward();
    }
  }

  public void backprop(Example example)
  {
    for (int i = 0; i < size(); ++i) {
      nodes[i].backprop(isOutputLayer ? example.expected[i] : 0.0);
    }
  }

  public void updateParams(double learningRate, double lambda, int trainSize, int batchSize)
  {
    for (Node node : nodes) {
      node.updateParams(learningRate, lambda, trainSize, batchSize);
    }
  }

  public double getCost(Example example)
  {
    if (!isOutputLayer) {
      throw new UnsupportedOperationException("Can't get cost from an internal (non-output) layer.");
    }

    double cost = 0.0;
    for (int i = 0; i < size(); ++i) {
      cost += network.costFunction.f(nodes[i].activation, example.expected[i]);
    }
    return cost;
  }
}
