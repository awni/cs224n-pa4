package cs224n.deep;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;

import org.ejml.ops.CommonOps;
import org.ejml.simple.SimpleMatrix;

public class WindowModel_2HL {
	// Parameters
	protected SimpleMatrix W1, W2, U;
	protected SimpleMatrix b1, b2, b3;
	protected SimpleMatrix LC;
	
	//
	public int windowSize, wordSize, hiddenSize, inputSize;
	double learningRate, C;
	int UUUNKKK = 0;
	String START = "<s>";
	String END = "</s>";
	String W1name = "W1.txt";
	String W2name = "W2.txt";
	String b1name = "b1.txt";
	String b2name = "b2.txt";
	String Uname = "U.txt";
	String b3name = "b3.txt";
	String Lname = "L.txt";
	String LCname = "LC.txt";

	public WindowModel_2HL(int _windowSize, int _hiddenSize, double _lr, double _reg) {
		this.windowSize = _windowSize;
		this.hiddenSize = _hiddenSize;
		this.wordSize = 50;
		this.inputSize = wordSize+5;
		this.learningRate = _lr;
		this.C = _reg;
		// TODO
			
	}

	/**
	 * Initializes the weights randomly.
	 */
	public void initWeights() {
		// TODO
		File fileE = new File(W1name);
		if(!fileE.exists()){
			
			double root6 = Math.sqrt(6);
			double wEInit = root6 / Math.sqrt(inputSize * windowSize + hiddenSize);
			double wEInit2 = root6 / Math.sqrt(hiddenSize + hiddenSize);
			double uEInit = root6 / Math.sqrt(hiddenSize + 1);
			
			
	
			W1 = SimpleMatrix.random(hiddenSize, inputSize * windowSize, -wEInit,
					wEInit, new Random());
			b1 = new SimpleMatrix(hiddenSize, 1); // initialized to 0s
			
			W2 = SimpleMatrix.random(hiddenSize, hiddenSize, -wEInit2, wEInit2, new Random());
			b2 = new SimpleMatrix(hiddenSize, 1); // initialized to 0s
					
			U = SimpleMatrix.random(1, hiddenSize, -uEInit, uEInit, new Random());
			b3 = new SimpleMatrix(1, 1); // initialized to 0s
	
			LC = SimpleMatrix.random(5,4,-.001,.001, new Random());
		}else{
			try {
				loadParams();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		// initialize with bias inside as the last column
		// W = SimpleMatrix...
		// U for the score
		// U = SimpleMatrix...
	}
	
	public double test(List<Datum> testData) throws IOException{
		// TODO
		int m = testData.size();
		double numCorrect, numReturned, numGold;
		numCorrect = numReturned = numGold = 0;
		
		File dev_result = new File("dev_results");
		BufferedWriter out = new BufferedWriter(new FileWriter(dev_result));
		
		for(int i=0; i<m; i++){
			if(testData.get(i).word.equals(START)){
				out.write("\n");
				continue;
			}
			SimpleMatrix x = getWindowedSample(testData, i);
			SimpleMatrix a1 = tanh((W1.mult(x)).plus(b1));
			SimpleMatrix a2 = tanh((W2.mult(a1)).plus(b2));
			double h = sigmoid(U.mult(a2).plus(b3)).get(0, 0);
			out.write(testData.get(i).word);
			
			//TODO comment before submit. Writes ground truth for perl script.
			if(getLabel(testData,i)==1){
				out.write("\tPERSON");
			}else{
				out.write("\tO");
			}

			if(h > 0.5){
				numReturned++;
				out.write("\tPERSON\n");
				if(getLabel(testData,i)==1){
					numGold++;
					numCorrect++;
				}
			}else{
				out.write("\tO\n");
				if(getLabel(testData,i)==1){
					numGold++;
				}
			}
			
		}
		out.close();
		double precision = numCorrect/numReturned;
		double recall = numCorrect/numGold;
		
		double F1 = 2*(precision*recall)/(precision+recall);
		
		System.out.println("F1 score: "+F1);
		return F1;
	}

	/**
	 * Simplest SGD training
	 */
	
	//TODO remove test param and remove testing of data in this function
	public void train(List<Datum> _trainData, List<Datum> testData) {
		// Setup -------------
		int m = _trainData.size();

		// Forward propagates for given sampleNum
		List<Integer> rand = new ArrayList<Integer>(m);
		for(int i=0;i<m;i++){
			rand.add(i);
		}
		int itNum = 0;
		double F1 = 0;
		double currF1;
		for(int count=0; count<10; count++){
			Collections.shuffle(rand);
			if(count>6) learningRate=.001;
			if(count>7) learningRate=.0001;
			
		for(int sampleNum = 0; sampleNum < m; sampleNum++){
			if(itNum%100000==0) {
				System.out.println("Iter "+itNum);
			}
			itNum++;
			SimpleMatrix x = getWindowedSample(_trainData, rand.get(sampleNum));
			SimpleMatrix a1 = tanh((W1.mult(x)).plus(b1));
			SimpleMatrix a2 = tanh((W2.mult(a1)).plus(b2));
			SimpleMatrix h = sigmoid(U.mult(a2).plus(b3));
			// Calculate derivatives
	
			// Common stuff
			double y = getLabel(_trainData, rand.get(sampleNum));
			double djdh = h.get(0, 0) - y;
			SimpleMatrix a2prime = a2.elementMult(a2).scale(-1);
			CommonOps.add(a2prime.getMatrix(), 1);
			SimpleMatrix a1prime = a1.elementMult(a1).scale(-1);
			CommonOps.add(a1prime.getMatrix(), 1);
	
			// dj/dU
			SimpleMatrix djdU = a2.transpose().scale(djdh).plus(U.scale(C));
	
			// dj/db3
			double[][] doubledjdb3 = { { djdh } };
			SimpleMatrix djdb3 = new SimpleMatrix(doubledjdb3);
	
			// dj/db2
			SimpleMatrix djdb2 = U.transpose().elementMult(a2prime).scale(djdh);
			
			// dj/dW2
			SimpleMatrix djdW2 = djdb2.mult(a1.transpose()).plus(W2.scale(C));;
			
			// dj/db1
			SimpleMatrix djdb1 = W2.transpose().mult(djdb2).elementMult(a1prime);
	
			// dj/dW1
			SimpleMatrix djdW1 = djdb1.mult(x.transpose()).plus(W1.scale(C));
	
			// dj/dL
			SimpleMatrix djdL = W1.transpose().mult(djdb1);
			
			
			U = U.plus(djdU.scale(-learningRate));
			b3 = b3.plus(djdb3.scale(-learningRate));
			W2 = W2.plus(djdW2.scale(-learningRate));
			b2 = b2.plus(djdb2.scale(-learningRate));
			W1 = W1.plus(djdW1.scale(-learningRate));
			b1 = b1.plus(djdb1.scale(-learningRate));
			updateWindowedSample(_trainData, rand.get(sampleNum), djdL);



//			// run gradient check
//			SimpleMatrix numdL = calculateNumGrad(x,0,x,y,m);
//			SimpleMatrix numdW1 = calculateNumGrad(W1,1,x,y,m);
//			SimpleMatrix numdb1 = calculateNumGrad(b1,2,x,y,m);
//			SimpleMatrix numdW2 = calculateNumGrad(W2,3,x,y,m);
//			SimpleMatrix numdb2 = calculateNumGrad(b2,4,x,y,m);
//			SimpleMatrix numdU = calculateNumGrad(U, 5, x, y, m);
//			SimpleMatrix numdb3 = calculateNumGrad(b3,6,x,y,m);
//	
//			double norm = 0;
//			norm += matrixNorm(djdL, numdL);
//			norm += matrixNorm(djdW1, numdW1);
//			norm += matrixNorm(djdb1, numdb1);
//			norm += matrixNorm(djdW2, numdW2);
//			norm += matrixNorm(djdb2, numdb2);
//			norm += matrixNorm(djdU, numdU);
//			norm += matrixNorm(djdb3, numdb3);
//			System.out.println("Norm is:" +Math.sqrt(norm));
		}
		System.out.println("Train data ");
		try {
			test(_trainData);
			System.out.println("Test data ");
			currF1 = test(testData);
			if(currF1 > F1){
				writeParams();
				F1 = currF1;
			}
		} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
		}
		}
		

	}
	
	private void loadParams() throws IOException{
		W1 = SimpleMatrix.loadCSV(W1name);
		W2 = SimpleMatrix.loadCSV(W2name);
		b1 = SimpleMatrix.loadCSV(b1name);
		b2 = SimpleMatrix.loadCSV(b2name);
		U = SimpleMatrix.loadCSV(Uname);
		b3 = SimpleMatrix.loadCSV(b3name);
		FeatureFactory.allVecs = SimpleMatrix.loadCSV(Lname);
		LC = SimpleMatrix.loadCSV(LCname);
	}
	
	private void writeParams() throws IOException{
		W1.saveToFileCSV(W1name);
		W2.saveToFileCSV(W2name);
		b1.saveToFileCSV(b1name);
		b2.saveToFileCSV(b2name);
		U.saveToFileCSV(Uname);
		b3.saveToFileCSV(b3name);
		FeatureFactory.allVecs.saveToFileCSV(Lname);
		LC.saveToFileCSV(LCname);
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
					c1 = cost(plus, W1, b1, W2, b2, U, b3, y, m);
					c2 = cost(minus, W1, b1, W2, b2, U, b3, y, m);
					break;
				case 1:
					c1 = cost(x, plus, b1, W2, b2, U, b3, y, m);
					c2 = cost(x, minus, b1, W2, b2, U, b3, y, m);
					break;
				case 2:
					c1 = cost(x, W1, plus, W2, b2, U, b3, y, m);
					c2 = cost(x, W1, minus, W2, b2, U, b3, y, m);
					break;
				case 3:
					c1 = cost(x, W1, b1, plus, b2, U, b3, y, m);
					c2 = cost(x, W1, b1, minus, b2, U, b3, y, m);
					break;
				case 4:
					c1 = cost(x, W1, b1, W2, plus, U, b3, y, m);
					c2 = cost(x, W1, b1, W2, minus, U, b3, y, m);
					break;
				case 5:
					c1 = cost(x, W1, b1, W2, b2, plus, b3, y, m);
					c2 = cost(x, W1, b1, W2, b2, minus, b3, y, m);
					break;
				case 6:
					c1 = cost(x, W1, b1, W2, b2, U, plus, y, m);
					c2 = cost(x, W1, b1, W2, b2, U, minus, y, m);
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

	private double cost(SimpleMatrix x, SimpleMatrix W1, SimpleMatrix b1, SimpleMatrix W2, SimpleMatrix b2,
			SimpleMatrix U, SimpleMatrix b3, double y, double m) {
		SimpleMatrix a1 = tanh((W1.mult(x)).plus(b1));
		SimpleMatrix a2 = tanh((W2.mult(a1)).plus(b2));
		double h = sigmoid(U.mult(a2).plus(b3)).get(0, 0);
		double cost = -y * Math.log(h) - (1 - y) * Math.log(1 - h);
		// with regulartization term
		cost = cost
				+ (C / (2))
				* (W1.elementMult(W1).elementSum() +W2.elementMult(W2).elementSum()+ U.elementMult(U)
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
	// Builds windowed input matrix
	private SimpleMatrix getWindowedSampleSentence(List<Datum> data, int sampleNum) {
		int m = data.size();
		String sample, prevSample;
		prevSample = null;
		int range = (windowSize - 1) / 2;
		int LCRows = LC.numRows();

		SimpleMatrix windowSample = new SimpleMatrix(windowSize * (wordSize+LCRows), 1);
		int cap = 0;

		// Make input mat with correct window size
		for (int i = 1; i <= range; i++) {
			
			if ((sampleNum - i) < 0) {
				sample = START;
			} else {
				sample = data.get(sampleNum - i).word;
				cap = getCapitalization(sample);
				sample = sample.toLowerCase();
			}
			
			if(sample.equals('.')){
				sample = START;
			}
			if(prevSample!=null && prevSample.equals(START)){
				sample = START;
			}
			Integer vecNum = FeatureFactory.wordToNum.get(sample);
			
			// If word doesn't exist insert UUUNKK
			if (vecNum == null)
				vecNum = UUUNKKK;

			SimpleMatrix capVec = LC.extractVector(false, cap);
			SimpleMatrix wordVec = FeatureFactory.allVecs.extractVector(false,
					vecNum);
			windowSample.insertIntoThis((range-i)*(wordSize+LCRows), 0, capVec);
			windowSample.insertIntoThis((range-i)*(wordSize+LCRows)+LCRows, 0, wordVec);
			prevSample = sample;
		}
		
		
		sample = data.get(sampleNum).word;
		cap = getCapitalization(sample);
		sample = sample.toLowerCase();
		Integer num = FeatureFactory.wordToNum.get(sample);
		if (num == null)
			num = UUUNKKK;
		SimpleMatrix capV = LC.extractVector(false, cap);
		SimpleMatrix wordV = FeatureFactory.allVecs.extractVector(false,
				num);
		windowSample.insertIntoThis((range)*(wordSize+LCRows), 0, capV);
		windowSample.insertIntoThis((range)*(wordSize+LCRows)+LCRows, 0, wordV);

		
		for (int i = 1; i <= range; i++) {
			
			if ((sampleNum + i) >= m) {
				sample = END;
			} else {
				sample = data.get(sampleNum).word;
				cap = getCapitalization(sample);
				sample = sample.toLowerCase();			}
			
			if(sample.equals('.')){
				sample = END;
			}
			if(prevSample.equals(END)){
				sample = END;
			}
			Integer vecNum = FeatureFactory.wordToNum.get(sample);
			
			// If word doesn't exist insert UUUNKK
			if (vecNum == null)
				vecNum = UUUNKKK;

			SimpleMatrix capVec = LC.extractVector(false, cap);
			SimpleMatrix wordVec = FeatureFactory.allVecs.extractVector(false,
					vecNum);
			windowSample.insertIntoThis((range+i)*(wordSize+LCRows), 0, capVec);
			windowSample.insertIntoThis((range+i)*(wordSize+LCRows)+LCRows, 0, wordVec);
			prevSample = sample;
		}
		
		
		return windowSample;
	}
	
	// Builds windowed input matrix
	private void updateWindowedSampleSentence(List<Datum> data, int sampleNum, SimpleMatrix djdL) {
		int m = data.size();
		String sample, prevSample;
		prevSample = null;
		int range = (windowSize - 1) / 2;
		int cap = 0;
		int LCRows = LC.numRows();
		// Make input mat with correct window size
		for (int i = 1; i <= range; i++) {
			if ((sampleNum - i) < 0) {
				sample = START;
			} else {
				sample = data.get(sampleNum).word;
				cap = getCapitalization(sample);
				sample = sample.toLowerCase();			}
			if(sample.equals('.')){
				sample = START;
			}
			if(prevSample!=null && prevSample.equals(START)){
				sample = START;
			}
			Integer vecNum = FeatureFactory.wordToNum.get(sample);
			
			// If word doesn't exist insert UUUNKK
			if (vecNum == null)
				vecNum = UUUNKKK;
			
			SimpleMatrix wordVec = FeatureFactory.allVecs.extractVector(false, vecNum);
			SimpleMatrix capVec = LC.extractVector(false, cap);
			
			SimpleMatrix updateCapVec = djdL.extractMatrix((range-i) * (wordSize+LCRows),(range-i) * (wordSize+LCRows)+LCRows,0,1);
			SimpleMatrix updateWordVec = djdL.extractMatrix((range-i) * (wordSize+LCRows)+LCRows,(range-i) * (wordSize+LCRows)+LCRows+wordSize,0,1);
			
			capVec = capVec.plus(updateCapVec.scale(-learningRate));
			wordVec = wordVec.plus(updateWordVec.scale(-learningRate));
			
			FeatureFactory.allVecs.insertIntoThis(0, vecNum, wordVec);
			LC.insertIntoThis(0, cap, capVec);
			prevSample = sample;

		}
		
		sample = data.get(sampleNum).word;
		cap = getCapitalization(sample);
		sample = sample.toLowerCase();		Integer num = FeatureFactory.wordToNum.get(sample);
		if (num == null)
			num = UUUNKKK;
		SimpleMatrix wordV = FeatureFactory.allVecs.extractVector(false, num);
		SimpleMatrix capV = LC.extractVector(false, cap);
		
		SimpleMatrix updateCapV = djdL.extractMatrix((range) * (wordSize+LCRows),(range) * (wordSize+LCRows)+LCRows,0,1);
		SimpleMatrix updateWordV = djdL.extractMatrix((range) * (wordSize+LCRows)+LCRows,(range) * (wordSize+LCRows)+LCRows+wordSize,0,1);
		
		capV = capV.plus(updateCapV.scale(-learningRate));
		wordV = wordV.plus(updateWordV.scale(-learningRate));
		
		FeatureFactory.allVecs.insertIntoThis(0, num, wordV);
		LC.insertIntoThis(0, cap, capV);
		
		for (int i = 1; i <= range; i++) {
			if ((sampleNum + i) >= m) {
				sample = END;
			} else {
				sample = data.get(sampleNum).word;
				cap = getCapitalization(sample);
				sample = sample.toLowerCase();
			}
			if(sample.equals('.')){
				sample = END;
			}
			if(prevSample.equals(END)){
				sample = END;
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