package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W, Wout;
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
		// initialize with bias inside as the last column
		// W = SimpleMatrix...
		// U for the score
		// U = SimpleMatrix...
	}


	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData ){
		int m = _trainData.size();
		
		//Input matrix
		SimpleMatrix input = new SimpleMatrix(windowSize*wordSize, m);
		
		//Target matrix
		SimpleMatrix target = new SimpleMatrix(1,m);
		
		buildInputAndTargetMatrix(_trainData, input, target);
		
		//	TODO
	}

	
	public void test(List<Datum> testData){
		// TODO
	}
	
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
