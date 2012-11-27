package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.ops.CommonOps;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

	//Parameters
	protected SimpleMatrix L, W, U;
	protected SimpleMatrix b1, b2;

	//
	public int windowSize, wordSize, hiddenSize;
	double learningRate, C;
	int UUUNKKK = 0;
	String START = "<s>";
	String END = "</s>";
	
	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		this.windowSize = _windowSize;
		this.hiddenSize = _hiddenSize;
		this.wordSize = 50;
		this.learningRate = _lr;
		this.C = 0.0;
		//TODO
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		//TODO
		
		double root6 = Math.sqrt(6);
		double wEInit = root6/Math.sqrt(wordSize*windowSize+hiddenSize);
		double uEInit = root6/Math.sqrt(hiddenSize+1);

		W = SimpleMatrix.random(hiddenSize, wordSize*windowSize, -wEInit, wEInit, new Random());
		b1 = new SimpleMatrix(hiddenSize, 1); //initialized to 0s

		U = SimpleMatrix.random(1, hiddenSize, -uEInit, uEInit, new Random());
		b2 = new SimpleMatrix(1,1); //initialized to 0s

		// initialize with bias inside as the last column
		// W = SimpleMatrix...
		// U for the score
		// U = SimpleMatrix...
	}


	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData ){
		//Setup -------------
		int m = _trainData.size();
		
		//Forward propagates for given sampleNum
		
		int sampleNum = 0; //temporary index until we setup sgd
		
		SimpleMatrix x = getWindowedSample(_trainData, sampleNum);
		SimpleMatrix a = tanh((W.mult(x)).plus(b1));
		SimpleMatrix h = sigmoid(U.mult(a).plus(b2));
		
		// Calculate derivatives
		
		// Common stuff
		double djdh = getLabel(_trainData, sampleNum) - h.get(0,0);
		SimpleMatrix aprime = a.elementMult(a).scale(-1);
		CommonOps.add(aprime.getMatrix(), 1);
		
		// dj/dU
		SimpleMatrix djdU = a.scale(djdh).plus(U.transpose().scale(C/m));

		// dj/db2
		double[][] doubledjdb2 = {{djdh}};
		SimpleMatrix djdb2 = new SimpleMatrix(doubledjdb2); 
		
		//dj/db1
		SimpleMatrix djdb1 = U.transpose().elementMult(aprime).scale(djdh);

		// dj/dW
		SimpleMatrix djdW = djdb1.mult(x.transpose()).plus(W.scale(C/m));
		
		// dj/dL
		SimpleMatrix djdL = W.transpose().mult(djdb1);
		
	}

	
	public void test(List<Datum> testData){
		// TODO
	}
	
	//Sigmoid: returns a new simple matrix with each element
	//of mat under the sigmoid function
	private SimpleMatrix sigmoid(SimpleMatrix mat){
		int numRows = mat.numRows();
		int numCols = mat.numCols();
		SimpleMatrix sig = new SimpleMatrix(numRows, numCols);
		
		for(int row=0; row<numRows; row++){
			for(int col=0; col<numCols; col++){
				double sigVal = mat.get(row,col);
				sigVal = 1/(1+Math.exp(-sigVal)); //sigmoid function
				sig.set(row,col,sigVal);
			}
		}
		
		return sig;
	}
	
	
	//Tanh: returns a new simple matrix with each element
	//of mat under the tanh function
	private SimpleMatrix tanh(SimpleMatrix mat){
		int numRows = mat.numRows();
		int numCols = mat.numCols();
		SimpleMatrix tan = new SimpleMatrix(numRows, numCols);
		
		for(int row=0; row<numRows; row++){
			for(int col=0; col<numCols; col++){
				double tanVal = mat.get(row,col);
				double pexp = Math.exp(tanVal);
				double nexp = Math.exp(-tanVal);
				tanVal =  (pexp-nexp)/(pexp+nexp);
				tan.set(row,col,tanVal);
			}
		}
		
		return tan;
	}
	
	//Gets integer label for sample num
	private int getLabel(List<Datum> data, int sampleNum){
		if(data.get(sampleNum).label.equals("O"))
			return 0;
		
		return 1;
	}
	
	//Builds windowed input matrix
	private SimpleMatrix getWindowedSample(List<Datum> data, int sampleNum){		
		
		int m = data.size();			
		String sample;
		int range = (windowSize-1)/2;
		SimpleMatrix windowSample = new SimpleMatrix(windowSize*wordSize,1);
		//Make input mat with correct window size
		for(int i=-range; i<=range; i++){
			
			
			if((sampleNum+i)<0){
				sample = START;
			}else if((sampleNum+i)>=m){
				sample = END;
			}else{
				sample = data.get(sampleNum+i).word;
			}
			
			Integer vecNum = FeatureFactory.wordToNum.get(sample);

			//If word doesn't exist insert UUUNKK
			if(vecNum == null)
				vecNum = UUUNKKK;
			
			SimpleMatrix wordVec = FeatureFactory.allVecs.extractVector(false, vecNum);
			windowSample.insertIntoThis((i+range)*wordSize, 0, wordVec);
		}
		return windowSample;
	}
	
}
