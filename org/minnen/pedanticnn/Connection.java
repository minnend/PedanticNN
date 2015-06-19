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
    // this.weight = NeuralNetwork.rng.nextGaussian() * weightScale; TODO
    this.weight = (parent.layer.index + parent.index + kid.index) * 0.1 + 0.1;
  }
  
  public String name() {
    return String.format("%s-%s", parent.name(), kid.name());
  }

  @Override
  public String toString() {
    return String.format("[Connection [%s]: w=%.3f  grad=%.5f (%.5f)]",
        name(), weight, gradWeight, fdWeight);
  }
}
