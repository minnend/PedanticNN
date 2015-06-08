package org.minnen.pedanticnn;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LearnNN
{
  private static List<Example> LoadData(String filename) throws IOException
  {
    List<Example> examples = new ArrayList<Example>();

    try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
      reader.readLine(); // skip header

      String line;
      while ((line = reader.readLine()) != null) {
        String[] fields = line.split(",");
        int label = Integer.parseInt(fields[0]);
        double[] data = new double[fields.length - 1];
        for (int i = 0; i < data.length; ++i) {
          int v = Integer.parseInt(fields[i + 1]);          
          data[i] = v / 255.0;          
        }
        examples.add(new Example(label, data));
      }
    }

    return examples;
  }
  
  private static void evaluate(int[] preds, List<Example> examples) {
    int nc = 0, n = examples.size();    
    for (int i=0; i<n; ++i) {
      // System.err.printf("%d: %d vs %d\n", i, preds[i], examples.get(i).label);
      if (examples.get(i).label == preds[i]) ++nc;      
    }    
    System.out.printf("%d / %d = %.2f%%\n", nc,  n, 100.0 * nc / n);
  }

  public static void main(String[] args) throws IOException
  {
    String trainFile = args[0];
    List<Example> examples = LoadData(trainFile);
    final int D = examples.get(0).numDims();
    System.err.printf("Training example: %d @ %dD\n", examples.size(), D);

    NeuralNetwork network = new NeuralNetwork(new int[]{ D, 10 });
    double[] output = network.getOutput();    
    
    network.train(examples);
    int[] preds = network.predict(examples);       
    evaluate(preds, examples);    
  }
}
