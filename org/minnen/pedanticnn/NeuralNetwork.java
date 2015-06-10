package org.minnen.pedanticnn;

import java.util.Random;

import org.minnen.pedanticnn.cost.CostFunction;

public class NeuralNetwork
{
  public static final Random rng = new Random();

  public final NeuralLayer[] layers;
  public CostFunction        costFunction;

  public NeuralNetwork(int[] layerSizes, CostFunction costFunction)
  {
    this.costFunction = costFunction;
    layers = new NeuralLayer[layerSizes.length];
    for (int i = 0; i < layers.length; ++i) {
      layers[i] = new NeuralLayer(i, layerSizes[i], i == layers.length - 1, this);
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

  public void train(Dataset dataTrain, Dataset dataTest, double learningRate, double lambda, int sizeMiniBatch,
      int numEpochs)
  {
    final int N = dataTrain.size();
    int miniBatchIndex = 0;
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
      // Randomize training data.
      dataTrain.shuffleExamples();

      // Run mini-batches.
      resetForLearning();
      for (int i = 0; i < N; ++i) {
        backprop(dataTrain.get(i));
        if (i % sizeMiniBatch == sizeMiniBatch - 1 || i == N - 1) {
          updateParams(learningRate, lambda, N, sizeMiniBatch);
          EvalResult evalTrain = LearnNN.evaluate(predict(dataTrain), dataTrain);
          EvalResult evalTest = LearnNN.evaluate(predict(dataTest), dataTest);
          System.out.printf("%d.%d: Cost=%f   Accuracy: %.2f%%, %.2f%%\n", epoch + 1, miniBatchIndex, evalTrain.cost,
              evalTrain.accuracy(), evalTest.accuracy());
          resetForLearning();
          ++miniBatchIndex;
        }
      }
      miniBatchIndex = 0;
    }
  }

  private void resetForLearning()
  {
    for (NeuralLayer layer : layers) {
      layer.resetForLearning();
    }
  }

  private void backprop(Example example)
  {
    // Push example through network.
    feedForward(example);

    // Update each layer starting with the output.
    for (int l = layers.length - 1; l > 0; --l) {
      layers[l].backprop(example);
    }
  }

  private void updateParams(double learningRate, double lambda, int trainSize, int batchSize)
  {
    for (NeuralLayer layer : layers) {
      layer.updateParams(learningRate, lambda, trainSize, batchSize);
    }
  }

  public double feedForward(Example example)
  {
    // Initialize input layer.
    layers[0].setActivations(example.data);

    // Feed-forward for other layers.
    for (int l = 1; l < layers.length; ++l) {
      layers[l].feedForward();
    }

    return getOutputLayer().getCost(example);
  }

  public Prediction predict(Example ex)
  {
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

  public Prediction[] predict(Dataset dataset)
  {
    Prediction[] preds = new Prediction[dataset.size()];
    for (int i = 0; i < dataset.size(); ++i) {
      preds[i] = predict(dataset.get(i));
    }
    return preds;
  }
}
