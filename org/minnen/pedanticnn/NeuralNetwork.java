package org.minnen.pedanticnn;

import java.util.List;
import java.util.Random;

public class NeuralNetwork
{
  public static final Random rng = new Random();

  public final NeuralLayer[] layers;

  public NeuralNetwork(int[] layerSizes)
  {
    layers = new NeuralLayer[layerSizes.length];
    for (int i = 0; i < layers.length; ++i) {
      layers[i] = new NeuralLayer(i, layerSizes[i]);
    }

    // Create dense connections
    for (int l = 1; l < layers.length; ++l) {
      for (int i = 0; i < layers[l - 1].size(); ++i) {
        Node src = layers[l - 1].node(i);
        for (int j = 0; j < layers[l].size(); ++j) {
          Node dst = layers[l].node(j);
          src.connect(dst);
        }
      }
    }
  }

  public NeuralLayer getInputLayer()
  {
    return layers[0];
  }

  public NeuralLayer getOutputLayer()
  {
    return layers[layers.length - 1];
  }

  public double[] getOutput()
  {
    NeuralLayer outputLayer = getOutputLayer();
    double[] output = new double[outputLayer.size()];
    for (int i = 0; i < output.length; ++i) {
      output[i] = outputLayer.node(i).activation;
    }
    return output;
  }

  public void train(List<Example> examples)
  {
    // TODO
  }

  public void feedForward(Example example)
  {
    // Initialize input layer.
    layers[0].setActivations(example.data);

    // Feed-forward for other layers.
    for (int l = 1; l < layers.length; ++l) {
      layers[l].feedForward();
    }
  }

  public int predict(Example ex)
  {
    feedForward(ex);
    double[] output = getOutputLayer().getActivations();
    int iBest = 0;
    for (int i = 1; i < output.length; ++i) {
      if (output[i] > iBest)
        iBest = i;
    }
    return iBest;
  }

  public int[] predict(List<Example> examples)
  {
    int[] preds = new int[examples.size()];
    for (int i = 0; i < examples.size(); ++i) {
      preds[i] = predict(examples.get(i));
    }
    return preds;
  }
}
