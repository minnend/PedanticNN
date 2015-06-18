package org.minnen.pedanticnn;

public class Connection
{
  public final Node parent, kid;

  public double     weight;
  public double     gradWeight;
  public double     fdWeight;  

  public Connection(Node parent, Node kid, double weightScale)
  {
    this.parent = parent;
    this.kid = kid;
    this.weight = NeuralNetwork.rng.nextGaussian() * weightScale;
  }

}
