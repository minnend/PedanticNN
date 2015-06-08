package org.minnen.pedanticnn;

public class Connection
{
  public double weight;
  public final Node parent, kid;
  
  public Connection(Node parent, Node kid) {
    this.parent = parent;
    this.kid = kid;
    this.weight = NeuralNetwork.rng.nextGaussian(); 
  }
}
