package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;
public class NER {

  public static void main(String[] args) throws IOException {

    if (args.length < 2) {
      System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
      return;
    }
    
    int windowSize = 9;
    int hiddenSize = 200;
    double learningRate = 0.01;
    double regularization = 0.0;
    
    if(args.length > 5){
        regularization = Double.parseDouble(args[args.length-4]);
        windowSize = Integer.parseInt(args[args.length-3]);
        hiddenSize = Integer.parseInt(args[args.length-2]);
        learningRate = Double.parseDouble(args[args.length-1]);
    }
    // this reads in the train and test datasets
    List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
    List<Datum> testData = FeatureFactory.readTestData(args[1]);

    // read the train and test data
    // TODO: Implement this function (just reads in vocab and word vectors)
    FeatureFactory.initializeVocab("../data/vocab.txt");
    SimpleMatrix allVecs = FeatureFactory.readWordVectors("../data/wordVectors.txt");

    // initialize model
//    WindowModel model = new WindowModel(windowSize, hiddenSize, learningRate, regularization);
    WindowModel3 model = new WindowModel3(windowSize, hiddenSize, learningRate, regularization);
    model.initWeights();
    
    // TODO: Implement those two functions
    model.train(trainData, testData);
    System.out.print("Test ");
    model.test(testData);
  }
}