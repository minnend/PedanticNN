package org.minnen.pedanticnn;

public class NeuralLayer
{
  private final Node[] nodes;
  public final int     index;

  public NeuralLayer(int index, int numNodes)
  {
    this.index = index;
    nodes = new Node[numNodes];
    for (int i = 0; i < numNodes; ++i) {
      nodes[i] = new Node(this, i);
    }
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

  /** Calculate activations for nodes in this layer from previous layer. */
  public void feedForward()
  {
    for (Node node : nodes) {
      node.feedForward();
    }
  }
}
