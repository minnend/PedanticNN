package org.minnen.pedanticnn;

import java.util.Random;

import org.minnen.pedanticnn.cost.CostFunction;

public class NeuralNetwork {
  public static final Random rng = new Random();

  public final NeuralLayer[] layers;
  public CostFunction costFunction;

  public NeuralNetwork(int[] layerSizes, CostFunction costFunction) {
    this.costFunction = costFunction;
    layers = new NeuralLayer[layerSizes.length];
    for (int i = 0; i < layers.length; ++i) {
      boolean isInputLayer = (i == 0);
      boolean isOutputLayer = (i == layers.length - 1);
      layers[i] = new NeuralLayer(i, layerSizes[i], isInputLayer, isOutputLayer, this);
    }

    // Create dense connections
    for (int l = 1; l < layers.length; ++l) {
      double weightScale = 1.0 / Math.sqrt(layerSizes[l - 1]);
      for (int i = 0; i < layers[l - 1].size(); ++i) {
        Node src = layers[l - 1].node(i);
        for (int j = 0; j < layers[l].size(); ++j) {
          Node dst = layers[l].node(j);
          src.connect(dst, weightScale);
        }
      }
    }
  }

  public NeuralLayer getInputLayer() {
    return layers[0];
  }

  public NeuralLayer getOutputLayer() {
    return layers[layers.length - 1];
  }

  public double[] getOutput() {
    NeuralLayer outputLayer = getOutputLayer();
    double[] output = new double[outputLayer.size()];
    for (int i = 0; i < output.length; ++i) {
      output[i] = outputLayer.node(i).activation;
    }
    return output;
  }

  public void train(Dataset dataTrain, Dataset dataTest, double learningRate, double lambda,
      int sizeMiniBatch, boolean checkGradients, int numEpochs) {
    final int N = dataTrain.size();
    int miniBatchIndex = 0;
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
      // Randomize training data.
      // dataTrain.shuffleExamples(); TODO

      // Run mini-batches.
      resetForLearning();
      for (int i = 0; i < N; ++i) {
        backprop(dataTrain.get(i));
        if (checkGradients) {
          checkGradients(dataTrain.get(i));
        }
        if (i % sizeMiniBatch == sizeMiniBatch - 1 || i == N - 1) {
          updateParams(learningRate, lambda, N, sizeMiniBatch);
          EvalResult evalTrain = LearnNN.evaluate(predict(dataTrain), dataTrain);
          EvalResult evalTest = LearnNN.evaluate(predict(dataTest), dataTest);
          System.out.printf("%d.%d: Cost=%f   Accuracy: %.2f%%, %.2f%%\n", epoch + 1,
              miniBatchIndex, evalTrain.cost, evalTrain.accuracy(), evalTest.accuracy());
          resetForLearning();
          ++miniBatchIndex;
        }
      }
      miniBatchIndex = 0;
    }
  }

  private void checkGradients(Example example) {
    System.out.println("Check gradients: " + example);
    int numBad = 0;
    for (NeuralLayer layer : layers) {
      if (layer.isInputLayer) { // no bias terms & no parent connections
        continue;
      }
      for (int i = 0; i < layer.size(); ++i) {
        Node node = layer.node(i);
        if (!checkGradientForBias(node, example)) {
          ++numBad;
        }

        for (Connection c : node.parents) {
          if (!checkGradientForWeight(c, example)) {
            ++numBad;
          }
        }
      }
    }

    System.out.println("After Back-Propagation:");
    System.out.println(dumpNetwork());

    if (numBad > 0) {
      throw new ArithmeticException(String.format(
          "Analytic derivative does not match"
          + " finite difference estimate (%d issues)",
          numBad));
    }
  }

  private boolean checkGradientForBias(Node node, Example example) {
    double delta = 1e-6;
    double eps = 0.001;

    double bias = node.bias;

    // Backward calculation: x - delta.
    node.bias -= delta;
    double backwardCost = feedForward(example);

    // Forward calculation: x + delta.
    node.bias = bias + delta;
    double forwardCost = feedForward(example);

    // Set bias back to original value.
    node.bias = bias;

    // Accumulate finite difference.
    node.fdBias += (forwardCost - backwardCost) / (2.0 * delta);

    // Verify gradients match.
    double sumAbs = Math.abs(node.gradBias) + Math.abs(node.fdBias);
    if (sumAbs > 1e-9) { // if sum is zero, we know we're ok and want to avoid div by zero
      double absError = Math.abs(node.gradBias - node.fdBias);
      double relError = absError / sumAbs;
      if (relError > eps) {
        System.err.printf("%s: %f vs %f  (absE=%f, relE=%f)\n", node.name(), node.gradBias,
            node.fdBias, absError, relError);
        return false;
      }
    }

    return true;
  }

  private boolean checkGradientForWeight(Connection c, Example example) {
    double delta = 1e-6;
    double eps = 0.001;

    double weight = c.weight;

    // Backward calculation: x - delta.
    c.weight -= delta;
    double backwardCost = feedForward(example);

    // Forward calculation: x + delta.
    c.weight = weight + delta;
    double forwardCost = feedForward(example);

    // Set weight back to original value.
    c.weight = weight;

    // Accumulate finite difference.
    c.fdWeight += (forwardCost - backwardCost) / (2.0 * delta);

    // Verify gradients match.
    double sumAbs = Math.abs(c.gradWeight) + Math.abs(c.fdWeight);
    if (sumAbs > 1e-9) { // if sum is zero, we know we're ok and want to avoid div by zero
      double absError = Math.abs(c.gradWeight - c.fdWeight);
      double relError = absError / sumAbs;
      if (relError > eps) {
        System.err.printf("[%s]: %f vs %f  (absE=%f, relE=%f)\n", c.name(), c.gradWeight,
            c.fdWeight, absError, relError);
        return false;
      }
    }

    return true;
  }

  private void resetForLearning() {
    for (NeuralLayer layer : layers) {
      layer.resetForLearning();
    }
  }

  private void backprop(Example example) {
    // Push example through network.
    feedForward(example);
    System.out.println("After Feed-Forward:");
    System.out.println(dumpNetwork());

    // Update each layer starting with the output.
    for (int l = layers.length - 1; l > 0; --l) {
      layers[l].backprop(example);
    }
  }

  private void updateParams(double learningRate, double lambda, int trainSize, int batchSize) {
    for (NeuralLayer layer : layers) {
      layer.updateParams(learningRate, lambda, trainSize, batchSize);
    }
  }

  public double feedForward(Example example) {
    // Initialize input layer.
    layers[0].setActivations(example.data);

    // Feed-forward for other layers.
    for (int l = 1; l < layers.length; ++l) {
      layers[l].feedForward();
    }

    return getOutputLayer().getCost(example);
  }

  public Prediction predict(Example ex) {
    double cost = feedForward(ex);
    double[] output = getOutputLayer().getActivations();
    int iBest = 0;
    for (int i = 1; i < output.length; ++i) {
      if (output[i] > output[iBest]) {
        iBest = i;
      }
    }
    return new Prediction(iBest, output, cost);
  }

  public Prediction[] predict(Dataset dataset) {
    Prediction[] preds = new Prediction[dataset.size()];
    for (int i = 0; i < dataset.size(); ++i) {
      preds[i] = predict(dataset.get(i));
    }
    return preds;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < layers.length; ++i) {
      sb.append(String.format("%d", layers[i].size()));
      if (i < layers.length - 1) {
        sb.append(",");
      }
    }
    return String.format("[NN: %s]", sb.toString());
  }

  public String dumpNetwork() {
    StringBuilder sb = new StringBuilder();

    for (int layerIndex = 0; layerIndex < layers.length; ++layerIndex) {
      NeuralLayer layer = layers[layerIndex];
      for (int nodeIndex = 0; nodeIndex < layer.size(); ++nodeIndex) {
        Node node = layer.node(nodeIndex);
        System.out.println(node);
        for (Connection c : node.kids) {
          System.out.println(" " + c);
        }
      }
    }

    return sb.toString();
  }
}
