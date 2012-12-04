package cs224n.deep;

import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.ops.CommonOps;
import org.ejml.simple.*;

import java.text.*;

public class WindowModel4 {

	// Parameters
	protected SimpleMatrix W, U;
	protected SimpleMatrix b1, b2;
	protected SimpleMatrix LC;


	//
	public int windowSize, wordSize, hiddenSize, inputSize;
	double learningRate, C;
	int UUUNKKK = 0;
	String START = "<s>";
	String END = "</s>";

	public WindowModel4(int _windowSize, int _hiddenSize, double _lr, double _reg) {
		this.windowSize = _windowSize;
		this.hiddenSize = _hiddenSize;
		this.wordSize = 50;
		this.learningRate = _lr;
		this.inputSize = wordSize+5;
		this.C = _reg;
		// TODO
	}

	/**
	 * Initializes the weights randomly.
	 */
	public void initWeights() {
		// TODO

		double root6 = Math.sqrt(6);
		double wEInit = root6 / Math.sqrt(inputSize * windowSize + hiddenSize);
		double uEInit = root6 / Math.sqrt(hiddenSize + 1);

		W = SimpleMatrix.random(hiddenSize, inputSize * windowSize, -wEInit,
				wEInit, new Random());
		b1 = new SimpleMatrix(hiddenSize, 1); // initialized to 0s

		U = SimpleMatrix.random(1, hiddenSize, -uEInit, uEInit, new Random());
		b2 = new SimpleMatrix(1, 1); // initialized to 0s

		LC = SimpleMatrix.random(5,4,-.001,.001, new Random());

		// initialize with bias inside as the last column
		// W = SimpleMatrix...
		// U for the score
		// U = SimpleMatrix...
	}
	
	public void test(List<Datum> testData) {
		// TODO
		int m = testData.size();
		double numCorrect, numReturned, numGold;
		numCorrect = numReturned = numGold = 0;
		for(int i=0; i<m; i++){
			SimpleMatrix x = getWindowedSample(testData, i);
			SimpleMatrix a = tanh((W.mult(x)).plus(b1));
			double h = sigmoid(U.mult(a).plus(b2)).get(0,0);
			Integer vecNum = FeatureFactory.wordToNum.get(testData.get(i).word.toLowerCase());
			if(h > 0.5){
				numReturned++;
				if(getLabel(testData,i)==1){
					numGold++;
					numCorrect++;
				}//else{
				//if(vecNum==null) System.out.println("word in unkkk");
				//System.out.println(testData.get(i).word + " should not be a person");}
			}else if(getLabel(testData,i)==1){
				//if(vecNum==null) System.out.println("word in unkkk");
				//System.out.println(testData.get(i).word + " should be a person");
				numGold++;
			}
		}
		double precision = numCorrect/numReturned;
		double recall = numCorrect/numGold;
		
		double F1 = 2*(precision*recall)/(precision+recall);
		
		System.out.println("F1 score: "+F1);
	}

	/**
	 * Simplest SGD training
	 */
	public void train(List<Datum> _trainData, List<Datum> testData) {
		// Setup -------------
		int m = _trainData.size();

		// Forward propagates for given sampleNum
		List<Integer> rand = new ArrayList<Integer>(m);
		for(int i=0;i<m;i++){
			rand.add(i);
		}
		int itNum = 0;
		for(int count=0; count<7; count++){
			Collections.shuffle(rand);
//			if(count>0){
//			System.out.println("Train data ");
//			test(_trainData);
//		    System.out.println("Test data ");
//			test(testData);}
		for(int sampleNum = 0; sampleNum < m; sampleNum++){
			//if(itNum%100000==0) System.out.println("iter is "+itNum);
			itNum++;
			SimpleMatrix x = getWindowedSample(_trainData, rand.get(sampleNum));
			SimpleMatrix a = tanh((W.mult(x)).plus(b1));
			SimpleMatrix h = sigmoid(U.mult(a).plus(b2));
			// Calculate derivatives
	
			// Common stuff
			double y = getLabel(_trainData, rand.get(sampleNum));
			double djdh = h.get(0, 0) - y;
			SimpleMatrix aprime = a.elementMult(a).scale(-1);
			CommonOps.add(aprime.getMatrix(), 1);
	
			// dj/dU
			SimpleMatrix djdU = a.transpose().scale(djdh).plus(U.scale(C));
	
			// dj/db2
			double[][] doubledjdb2 = { { djdh } };
			SimpleMatrix djdb2 = new SimpleMatrix(doubledjdb2);
	
			// dj/db1
			SimpleMatrix djdb1 = U.transpose().elementMult(aprime).scale(djdh);
	
			// dj/dW
			SimpleMatrix djdW = djdb1.mult(x.transpose()).plus(W.scale(C));
	
			// dj/dL
			SimpleMatrix djdL = W.transpose().mult(djdb1);
			
			
			U = U.plus(djdU.scale(-learningRate));
			b2 = b2.plus(djdb2.scale(-learningRate));
			W = W.plus(djdW.scale(-learningRate));
			b1 = b1.plus(djdb1.scale(-learningRate));
			updateWindowedSample(_trainData, rand.get(sampleNum), djdL);


//			// run gradient check
//			SimpleMatrix numdL = calculateNumGrad(x,0,x,y,m);
//			SimpleMatrix numdW = calculateNumGrad(W,1,x,y,m);
//			SimpleMatrix numdb1 = calculateNumGrad(b1,2,x,y,m);
//			SimpleMatrix numdU = calculateNumGrad(U, 3, x, y, m);
//			SimpleMatrix numdb2 = calculateNumGrad(b2,4,x,y,m);
//			
//			double norm = 0;
//			norm += matrixNorm(djdL, numdL);
//			norm += matrixNorm(djdW, numdW);
//			norm += matrixNorm(djdb1, numdb1);
//			norm += matrixNorm(djdU, numdU);
//			norm += matrixNorm(djdb2, numdb2);
//			System.out.println("Norm is:" +Math.sqrt(norm));
		}
		  
		}
		System.out.print("Train ");
		test(_trainData);
	}

	private double matrixNorm(SimpleMatrix grad, SimpleMatrix numGrad){
		double norm=0.0;
		for(int i=0; i<grad.numRows(); i++){
			for(int j=0; j<grad.numCols(); j++){
				norm+=Math.pow(grad.get(i,j)-numGrad.get(i,j),2);
			}
		}
		return norm;
	}
	private SimpleMatrix calculateNumGrad(SimpleMatrix param, int paramNum,SimpleMatrix x, double y, double m) {
		double epsilon = .0001;
		double c1, c2;
		SimpleMatrix numGrad = new SimpleMatrix(param.numRows(),
				param.numCols());

		for (int i = 0; i < param.numRows(); i++) {
			for (int j = 0; j < param.numCols(); j++) {
				SimpleMatrix plus = new SimpleMatrix(param);
				plus.set(i, j, param.get(i, j) + epsilon);
				SimpleMatrix minus = new SimpleMatrix(param);
				minus.set(i, j, param.get(i, j) - epsilon);

				switch (paramNum) {
				case 0:
					c1 = cost(plus, W, b1, U, b2, y, m);
					c2 = cost(minus, W, b1, U, b2, y, m);
					break;
				case 1:
					c1 = cost(x, plus, b1, U, b2, y, m);
					c2 = cost(x, minus, b1, U, b2, y, m);
					break;
				case 2:
					c1 = cost(x, W, plus, U, b2, y, m);
					c2 = cost(x, W, minus, U, b2, y, m);
					break;
				case 3:
					c1 = cost(x, W, b1, plus, b2, y, m);
					c2 = cost(x, W, b1, minus, b2, y, m);
					break;
				case 4:
					c1 = cost(x, W, b1, U, plus, y, m);
					c2 = cost(x, W, b1, U, minus, y, m);
					break;
				default:
					c1 = c2 = 0.0;
					break;
				}
				numGrad.set(i, j, (c1 - c2) / (2 * epsilon));
			}
		}
		return numGrad;
	}

	private double cost(SimpleMatrix x, SimpleMatrix W, SimpleMatrix b1,
			SimpleMatrix U, SimpleMatrix b2, double y, double m) {
		SimpleMatrix a = tanh((W.mult(x)).plus(b1));
		double h = sigmoid(U.mult(a).plus(b2)).get(0, 0);
		double cost = -y * Math.log(h) - (1 - y) * Math.log(1 - h);
		// with regulartization term
		cost = cost
				+ (C / (2))
				* (W.elementMult(W).elementSum() + U.elementMult(U)
						.elementSum());
		return cost;
	}


	// Sigmoid: returns a new simple matrix with each element
	// of mat under the sigmoid function
	private SimpleMatrix sigmoid(SimpleMatrix mat) {
		int numRows = mat.numRows();
		int numCols = mat.numCols();
		SimpleMatrix sig = new SimpleMatrix(numRows, numCols);

		for (int row = 0; row < numRows; row++) {
			for (int col = 0; col < numCols; col++) {
				double sigVal = mat.get(row, col);
				sigVal = 1 / (1 + Math.exp(-sigVal)); // sigmoid function
				sig.set(row, col, sigVal);
			}
		}

		return sig;
	}

	// Tanh: returns a new simple matrix with each element
	// of mat under the tanh function
	private SimpleMatrix tanh(SimpleMatrix mat) {
		int numRows = mat.numRows();
		int numCols = mat.numCols();
		SimpleMatrix tan = new SimpleMatrix(numRows, numCols);

		for (int row = 0; row < numRows; row++) {
			for (int col = 0; col < numCols; col++) {
				double tanVal = mat.get(row, col);
				double pexp = Math.exp(tanVal);
				double nexp = Math.exp(-tanVal);
				tanVal = (pexp - nexp) / (pexp + nexp);
				tan.set(row, col, tanVal);
			}
		}

		return tan;
	}

	// Gets integer label for sample num
	private double getLabel(List<Datum> data, int sampleNum) {
		if (data.get(sampleNum).label.equals("O"))
			return 0;

		return 1;
	}

	// Get capitalization
	private int getCapitalization(String word){
		boolean firstCapital = false;
		int numCap = 0;
		for(int i=0; i<word.length(); i++){
			char let = word.charAt(i);
			if(Character.isUpperCase(let)){
				numCap++;
				if(i==0)
					firstCapital = true;
			}
		}
			
		if(numCap==word.length())
			return 2;
		
		if(numCap==1 && firstCapital)
			return 1;
		
		if(numCap==0)
			return 0;
					
		return 3;
	}
	
	// Builds windowed input matrix
		private SimpleMatrix getWindowedSample(List<Datum> data, int sampleNum) {

			int m = data.size();
			String sample;
			int LCRows = LC.numRows();
			int range = (windowSize - 1) / 2;
			SimpleMatrix windowSample = new SimpleMatrix(windowSize * (wordSize+LCRows), 1);
			// Make input mat with correct window size
			for (int i = -range; i <= range; i++) {
				int cap = 0;
				if ((sampleNum + i) < 0) {
					sample = START;
				} else if ((sampleNum + i) >= m) {
					sample = END;
				} else {
					sample = data.get(sampleNum + i).word;
					cap = getCapitalization(sample);
					sample = sample.toLowerCase();
				}

				Integer vecNum = FeatureFactory.wordToNum.get(sample);

				// If word doesn't exist insert UUUNKK
				if (vecNum == null)
					vecNum = UUUNKKK;

				SimpleMatrix capVec = LC.extractVector(false, cap);
				SimpleMatrix wordVec = FeatureFactory.allVecs.extractVector(false,
						vecNum);
				windowSample.insertIntoThis((i+range)*(wordSize+LCRows), 0, capVec);
				windowSample.insertIntoThis((i+range)*(wordSize+LCRows)+LCRows, 0, wordVec);
			}
			return windowSample;
		}
		
		// Updates allVecs with djdL
		private void updateWindowedSample(List<Datum> data, int sampleNum, SimpleMatrix djdL) {

			int m = data.size();
			String sample;
			int range = (windowSize - 1) / 2;
			int LCRows = LC.numRows();

			for (int i = -range; i <= range; i++) {
				int cap = 0;
				if ((sampleNum + i) < 0) {
					sample = START;
				} else if ((sampleNum + i) >= m) {
					sample = END;
				} else {
					sample = data.get(sampleNum + i).word;
					cap = getCapitalization(sample);
					sample = sample.toLowerCase();
				}

				Integer vecNum = FeatureFactory.wordToNum.get(sample);

				// If word doesn't exist insert UUUNKK
				if (vecNum == null)
					vecNum = UUUNKKK;
				
				SimpleMatrix wordVec = FeatureFactory.allVecs.extractVector(false, vecNum);
				SimpleMatrix capVec = LC.extractVector(false, cap);
				
				SimpleMatrix updateCapVec = djdL.extractMatrix((i + range) * (wordSize+LCRows),(i + range) * (wordSize+LCRows)+LCRows,0,1);
				SimpleMatrix updateWordVec = djdL.extractMatrix((i + range) * (wordSize+LCRows)+LCRows,(i + range) * (wordSize+LCRows)+LCRows+wordSize,0,1);
				
				capVec = capVec.plus(updateCapVec.scale(-learningRate));
				wordVec = wordVec.plus(updateWordVec.scale(-learningRate));
				
				FeatureFactory.allVecs.insertIntoThis(0, vecNum, wordVec);
				LC.insertIntoThis(0, cap, capVec);

			}
		} 


}
