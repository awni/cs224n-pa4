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
	double learningRate;
	
	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		this.windowSize = _windowSize;
		this.hiddenSize = _hiddenSize;
		this.wordSize = 50;
		this.learningRate = _lr;
		
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
		//Input matrix
		SimpleMatrix input = new SimpleMatrix(windowSize*wordSize, m);
		//Target matrix
		SimpleMatrix target = new SimpleMatrix(1,m);
		buildInputAndTargetMatrix(_trainData, input, target);
		
		
		//Forward propagates for given sampleNum
		int sampleNum = 0;
		double C = 0.0;
		SimpleMatrix a = tanh((W.mult(input.extractVector(false, sampleNum))).plus(b1));
		SimpleMatrix h = sigmoid(U.mult(a).plus(b2));
		
		// Calculate derivatives
		
		// Common stuff
		double djdh = target.get(1, sampleNum) - h.get(1, 1);
		SimpleMatrix one = new SimpleMatrix(a.numRows(), a.numCols(), true, 1);
		one.print();
		SimpleMatrix aprime = one.plus(a.elementMult(a).scale(-1));
		
		// dj/dU
		SimpleMatrix djdU = a.scale(djdh).plus(U.scale(C/m));
		
		// dj/db(2)
		double[][] doubledjdb2 = {{djdh}};
		SimpleMatrix djdb2 = new SimpleMatrix(doubledjdb2); 
		
		
		
		//Backwards propagate
		
		//Run SGD calculating gradient then updating params 
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
	
	//Builds windowed input matrix and target matrix which contains labels for center word 
	//in each input window sample
	private void buildInputAndTargetMatrix(List<Datum> data, SimpleMatrix input, SimpleMatrix target){		
		int m = data.size();
		int UUUNKKK = 0;
		String START = "<s>";
		String END = "</s>";
		
		String sample;
		int range = (windowSize-1)/2;
		for(int i=0; i<m; i++){
			
			//Set label in target matrix
			String label = data.get(i).label;
			if(label.equals("O")){
				target.set(0,i,0);
			}else{
				target.set(0,i,1);
			}
			
			//Make input mat with correct window size
			for(int j=-range; j<=range; j++){
				Integer vecNum;
				
				if((i+j)<0){
					sample = START;
				}else if((i+j)>=m){
					sample = END;
				}else{
					sample = data.get(i+j).word;
				}
				
				vecNum = FeatureFactory.wordToNum.get(sample);

				//If word doesn't exist insert UUUNKK
				if(vecNum == null)
					vecNum = UUUNKKK;
				
				SimpleMatrix wordVec = FeatureFactory.allVecs.extractVector(true, vecNum);
				input.insertIntoThis((j+range)*wordSize, i, wordVec.transpose());
			}
		}
		
	}
	
}
