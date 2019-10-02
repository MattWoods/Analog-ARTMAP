/*
 *    AnalogARTMAP.java : Matthew Woods (mattw@cns.bu.edu)
 *
 *
 */
package weka.classifiers.functions;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Summarizable;
import weka.core.Utils;

/**
 * Class implementing Analog artmap (AnalogARTMAP) for regression.
 *
 *
 * @author Matthew Woods (mattw@cns.bu.edu)
 * @version $Revision: 1.0 $
 * $Id: AnalogARTMAP.java,v 1.0 3/1/06 19:26:47 matt Exp $
 */

public class AnalogARTMAP extends Classifier implements OptionHandler, Summarizable {


	private static final long serialVersionUID = -3714059771618599407L;

	double alpha =		0.001;	// CBD and Weber law signal parameter default: 0.01
	double p =		2.0;    	// CAM rule power default: 2
	double beta =		1.0;	// learning rate
	double epsilon =	-0.001;	// MT rule parameter. set as command line param -e.
	double rho_a_bar =	0.0;	// ARTa baseline vigilance
	double  t_u;     // F0 -> F2 signal to uncomited nodes.
	double node_threshold = 0.00; //only nodes firing above this level are used for classification
	double Q = 13;				// Q-max CAM rule parameter
	double rho_b = 0.2;		// Analagous to the vigilance parameter,
	//but this is for the analog outputs
	int number_of_voters = 1;	//number of Analog ARTMAP voters in ensemble
	String classifierFileName=new String("classifier.m");
	/** F0 node activations */
	private double [] A;

	/** F2 node activations */
	private double [] y;

	/** Associations betwen the coding nodes and ouptut nodes.
	 * These track the (mean) analog output values associated
	 * with each F2 node*/
	private double [] W;

	/** LTM weights from F0 to F2*/
	private double [][] w;

	/**Total features in the feature vector*/
	private int M;

	/** rho, vigilance parameter*/
	private double rho;

	/** C, the total number of committed nodes*/
	private int C;

	/** L, the total number of classes in data*/
	private int L;

	/**total training epochs*/
	private int m_numEpochs;

	/**cj, instance counting weights, here they are only used for the analog output
	 * layer learning, not for weighting of the predictions as in ARTMAPIC.
	 */
	private double [] cj;

	/** this line implements voting:
	 */
	private AnalogARTMAP [] artmaps;

	/**
	 * Builds AnalogARTMAP classifier
	 *
	 * @param data the training data
	 * @exception Exception if classifier can't be built successfully
	 */
	public void buildClassifier(Instances data) throws Exception {

//		these lines were also added to implement voting:
		if (number_of_voters > 1) {
			artmaps = new AnalogARTMAP [number_of_voters];
			for (int voterNumber = 0; voterNumber<(number_of_voters); voterNumber++) {

				Instances instcopy = new Instances (data);
				//The following line gives a new input ordering to each voter in the ensemble
				//in such a way that repeated runs with the same number of voters will have
				//the same random orderings:
				instcopy.randomize(new Random(voterNumber));
				//To give non-repeatably random orderings to the inputs comment the above line
				//and uncomment the following line:
				//instcopy.randomize(new Random());
				String [] opts = this.getOptions();
				artmaps[voterNumber] = new AnalogARTMAP();
				artmaps[voterNumber].setOptions(opts);
				artmaps[voterNumber].setNumberOfVoters(1);
				artmaps[voterNumber].buildClassifier(instcopy);

			}
			return;
		}


		Enumeration enumAtt = data.enumerateAttributes();
		while (enumAtt.hasMoreElements()) {
			Attribute attr = (Attribute) enumAtt.nextElement();
			Enumeration _enum = data.enumerateInstances();
			while (_enum.hasMoreElements()) {
				if (((Instance) _enum.nextElement()).isMissing(attr)) {
					throw new Exception("AnalogARTMAP: No missing values, please.");
				}
			}
		}


		data = initMap(data);

		//see if this makes the difference
		//rescaleRho_b(Instances data);

		makeMap(data);

	}

	private void rescaleRho_b(Instances data ) {
		Enumeration _enum = data.enumerateInstances();
		double analogOutputMin = Double.MAX_VALUE;
		double analogOutputMax = Double.MIN_VALUE;
		while (_enum.hasMoreElements()) {
			Instance localInst = (Instance) _enum.nextElement();
			double analogOutput = localInst.classValue();
			if (analogOutput > analogOutputMax){
				analogOutputMax = analogOutput;
			}
			if (analogOutput < analogOutputMin){
				analogOutputMin = analogOutput;
			}
		}
		rho_b *= (analogOutputMax - analogOutputMin);
	}



	/** Initialise the AnalogARTMAP by allocating storage for weights and
	 * activity pattern vectors. Also normalizes the data to [0,1]
	 * hypercube and returns it to caller.

	 @param data the training data.
	 @exception Exception if initialization cannot be performed successfully.*/

	private Instances initMap(Instances data) throws Exception {

		//activation vectors reset to zero. Double the length of total
		//attributes for complement coding

		System.err.println("Dataset has " + data.numAttributes() + " Attributes");
		M = data.numAttributes()-1;
		System.err.println("Dataset has " + data.numClasses() + " Classes");
		L = data.numClasses();
		System.err.println("Dataset has " + data.numInstances() + " Instances");

		data.deleteWithMissingClass();

		A = new double [2*M] ;
		C=0;

		return data;
	}

	/**
	 * Method building AnalogARTMAP.
	 *
	 * @param data the training data
	 * @exception Exception if AnalogARTMAP can't be built successfully
	 * @exception UnexpectedStateException if the learning state switch is broken.
	 */
	private void makeMap(Instances data){
		try{
			System.err.println("makemap");

			int currentStep = 1; //current learning step.
			double [] sigma;  //signal from F3 to F1

			int J=0;  //0 value just to remove a "might not have been
			//initialized" warning.
			double kPrime=0; //network estimate of analog output.

			int currentEpoch=1;
			System.err.println("Epoch " + currentEpoch);


			boolean previousWasLastSample = false;


			int n=1;
			data.randomize(new Random(1));
			Enumeration instEnum = (data).enumerateInstances();
			Instance currentInstance = (Instance) instEnum.nextElement();
			boolean keepworking = true;


			J=-1;
			while(keepworking)
			{


				switch(currentStep)
				{
				case 1:
				{
					//First iteration.

					activateF0(currentInstance);
					C=1;
					y=new double [1];
					w = new double [2*M][1];
					W = new double [1];
					//add the instance counting weight upadate:
					cj = new double [1];

					W[0]=(double) currentInstance.classValue();;
					y[0]=1;

					cj[0] = 1;

					//set the weights
					for(int i=0;i<2*M;i++)
						w[i][0]=A[i];

					//goto step 9, to choose the next sample.
					currentStep = 9;


				}
				break;

				case 2:
				{
					//reset.
					if (C==2);{
					double another_dummy_variable=0;
					}
					double [] T = computeSignal();
					J=-1;
					Hashtable delta = new Hashtable();

					//all the inactive nodes go in here first..
					for (int i=0;i<T.length;i++)
					{
						if(T[i]<alpha*M)
							delta.put(new Integer(i),new Integer(i));
					}


					//search for a coding node.. O(n^2)
					boolean truePrediction = false;
					while(delta.size()<T.length && !truePrediction)
					{

						double maxT=0;
						J=-1;
						for (int i=0;i<T.length;i++)
						{
							if(!delta.containsKey(new Integer(i)))
							{
								if(maxT<T[i])
								{
									maxT=T[i];
									J=i;
								}
							}
						}

						//at the end of the above loop, we have the largest T (ie, J)
						//matching criterion:
						if(J!=-1) //if there's at least one node above threshold
						{
							double temp_hack = l1Norm(min(A,getCol(w,J)));
							if(l1Norm(min(A,getCol(w,J)))>=M*rho) //vigilance passed?
							{


								//WTA coding.
								y=new double [y.length];
								y[J]=1;
								double hacked_out_dif =Math.abs(W[J] - (double)currentInstance.classValue());
								if (Math.abs(W[J] - (double)currentInstance.classValue())<= rho_b)
								{

									truePrediction=true;
									//	       System.err.println("resonance"+this+"\n J: " + J);
									double minWA[]=min(A,getCol(w,J));
									//update the instance counting weight of the winning node:

									cj[J]= cj[J] + 1;
									//////////////////////////////////
									//if (n == 21){
									//	double another_dummy = 0;
									//};
									if (cj[J] == 21){
										double another_dummy = 0;
									};
									///////////////////////////////////
									for(int i=0;i<w.length;i++)
									{
										//here is the learning for the w weights:
										w[i][J]=beta*minWA[i] + (1-beta)*w[i][J];
									}
									//here is the learning for the W weights:
									W[J] = W[J] + (1/cj[J])*((double)currentInstance.classValue() - W[J]);
									//}

								}
								else

								{
									delta.put(new Integer(J),new Integer(J));

									//raise vigilance.
									rho=l1Norm(min(A,getCol(w,J)))/M + epsilon;

								}
							}
							else //vigilance failure. add J to refraction.
							{
								delta.put(new Integer(J),new Integer(J));

							}
						}
						else
						{
							throw new Exception("Something wrong while looking for maximal T");
						}
					}

					if(!truePrediction) //if we came out of the above loop without
						//correctly predicting (aka, next stop is
						//vigilance), we need to add a node.
					{

						addUnit(currentInstance);
					}
				}

				case 9: //Next iteration.
				{
					if(previousWasLastSample) //if we saw the last sample earlier,
						//forget about more samples and get
						//out of while leoop
					{
						if(currentEpoch>=m_numEpochs)
						{
							keepworking=false; //we go out of this loop now.
							System.err.println("done..");
						}
						else //epochs are not over yet.
						{
							currentEpoch++;
							System.err.println("Epoch " + currentEpoch+ "Total categories: " + y.length);
							data.randomize(new Random(2));
							instEnum=data.enumerateInstances();
							currentInstance=(Instance) instEnum.nextElement();
							activateF0(currentInstance);
							rho=rho_a_bar;
							currentStep = 2;
							if(n % 100 == 0)
								System.err.print(".");
							previousWasLastSample=false;
						}
					}
					else
					{

						currentInstance= (Instance) instEnum.nextElement();
						activateF0(currentInstance);
						rho=rho_a_bar;

						currentStep = 2;
						if(n % 100 == 0)
							System.err.print(".");
						n++;
						//if (n == 22){
							//double put_break_point_here = 1;
						//};
						if(!instEnum.hasMoreElements())
							previousWasLastSample=true;


					}


				}

				break;

				default:

					throw new Exception("AnalogARTMAP: Something seriously wrong in the learning switch");

				}

			}


		}
		catch (Exception e)
		{
			System.err.println("Exception occured in makeMap "+e.getMessage() );
			e.printStackTrace();
		}
	}

	/** Computes the f0->f2 signal T
	 */

	private double [] computeSignal () throws Exception
	{
		//1D array of length C for computing T.
		double [] minAw= min(A,w);

		double [] w_j = l1Norm(w,0);

		double [] ans = new double[w_j.length];

		for (int i=0;i<w_j.length;i++)
			ans[i] = minAw[i]+(1-alpha)*(M-w_j[i]);


		return ans;
	}


	private double [] min(double [] a, double [] w) throws Exception
	{
		double [] T  = new double[a.length];

		for (int i=0;i<T.length;i++)
		{
			T[i]=(Math.min(a[i],w[i]));
		}

		return T;
	}

	private double [] abs(double [] a)
	{
		double [] ans = new double[a.length];
		for (int i=0;i<ans.length;i++)
		{
			ans[i]=Math.abs(a[i]);
		}
		return (ans);
	}


	/** returns min(A,w) in a 1D array of length C */
	private double [] min(double [] a,double [][] w) throws Exception
	{
		double [] T = new double[w[0].length];

		for (int i=0;i<a.length;i++)
		{
			for(int j=0;j<T.length;j++)
			{
				T[j]+=(Math.min(a[i],w[i][j]));
			}
		}
		return T;
	}


	private double l1Norm(double [] a) throws Exception
	{
		double sum = 0;
		for (int i=0;i<a.length;i++)
		{
			sum+=Math.abs(a[i]);
		}
		return sum;
	}



	/**L1 norm (sum(abs(i))) of array a along axis id given by axis*/
	private double [] l1Norm(double [][] a,int axis) throws Exception
	{
		double [] sum = new double[a.length];

		if(axis==0)
			sum= new double[a[0].length];

		for (int i=0;i<a.length;i++)
		{
			for (int j=0;j<a[i].length;j++)
			{
				if(axis==1) //sum along rows of tmp.
					sum[i]+=Math.abs(a[i][j]);
				else if(axis==0) //sum along columns of tmp.
					sum[j]+=Math.abs(a[i][j]);
				else throw new Exception("Invalid l1Norm axis id given");
			}
		}
		return (sum);
	}

	/** Activates the F0 nodes based on instance
	 *  @param currentInstance being presented to the system
	 */

	private void activateF0(Instance currentInstance)
	{

		try{

			A = new double[2*M];
			for (int i = 0,j=0; i<M;i++,j++)
			{

				if(j==currentInstance.classIndex())
					j++;

				if(!currentInstance.isMissing(j))
				{
					A[i]=(currentInstance.value(j));

					if(A[i] > 1)
						A[i]=1;
					else if (A[i]<0)
						A[i]=0;

					A[i+M]=1.0-A[i];

					if(Double.isNaN(A[i])||Double.isNaN(A[i+M]))
						throw new Exception("A["+i+"] or A[i+"+M+"] is NaN. Do something!");
				}

			}
		}
		catch (Exception e)
		{
			System.err.println("Exception in AnalogARTMAP.activateF0 " + e.getMessage());
			e.printStackTrace();
		}
	}



	private double [] getCol(double [][] w, int k)
	{
		double [] tmpw = new double[w.length];
		for(int i=0;i<tmpw.length;i++)
			tmpw[i]=w[i][k];
		return tmpw;
	}


	/**
	 * Classifies a given test instance using the AnalogARTMAP.
	 *
	 * @param instance the instance to be classified
	 * @return the classification
	 */
	public double classifyInstance(Instance currentInstance)  throws Exception{



		// Test Step 1: Test vector n:
		double  kPrime = 0;
		double[] kPrimes = new double[number_of_voters];
		if (number_of_voters > 1){
			double kPrimeTemp = 0;
			for (int voterNumber = 0; voterNumber<(number_of_voters); voterNumber++) {
				kPrime = artmaps[voterNumber].classifyInstance(currentInstance);
				kPrimes[voterNumber] = kPrime;
				kPrimeTemp += kPrime;
			}
			kPrime = kPrimeTemp/number_of_voters;
			return kPrime;
		}
		activateF0(currentInstance);
		y=new double [y.length];
		double [] T = computeSignal();


		int totalpointboxes=0;
		int lambdasize=0;

		for(int i=0;i<T.length;i++)
		{
			if(T[i]==M)
			{
				totalpointboxes++;
			}
			if(T[i]>alpha*M)
			{
				lambdasize++;
			}
		}

		//point box case
		if(totalpointboxes>0)
		{
			for(int i=0;i<y.length;i++)
			{
				if(T[i]==M)
					y[i]=(double) 1/totalpointboxes;
			}
		}
		else if (lambdasize>0) //
		{
			double denominator=0;
			for (int i=0;i<y.length;i++)
			{
				if(T[i]>alpha*M)
					denominator+=Math.pow(1/(M-T[i]),p);
			}
			for(int i=0;i<y.length;i++)
			{
				if(T[i]>alpha*M)
					y[i]=(double) Math.pow(1/(M-T[i]),p)/denominator;
			}
		}
		//Q-max rule:
		if (Q<C){
			double [] ycopy  = y;
			double [] yQd  = new double[y.length];


			for(int i=0;i<ycopy.length;i++){
				yQd[i]=0;
			}
			for (int q=0;q<Q;q++){
				double yHighest = 0;
				int yHighestLocation = -1;
				for(int i=0;i<ycopy.length;i++){
					if (ycopy[i]>yHighest){
						yHighest = ycopy[i];
						yHighestLocation=i;
					}
				}
				if (yHighestLocation>-1){
					yQd[yHighestLocation] = yHighest;
					ycopy[yHighestLocation] = 0;
				}
				else{
					if (q==0){
						System.err.println("y is equal to " + y);
						throw new Exception("AnalogARTMAP: Something seriously wrong in the Q-max rule implementation");
					}
				}
			}
			double yQdSum=0;
			for (int i=0;i<yQd.length;i++){
				yQdSum+=yQd[i];

			}
			if (yQdSum<=0){
				throw new Exception("AnalogARTMAP: y values have a non-positive sum.");
			}
			//renormalize y values after applying Q-max
			for (int i=0;i<y.length;i++){
				y [i]= yQd[i]/yQdSum;
			}
		}
///////////////////////////////////////

		for(int i=0;i<y.length;i++)
		{
			kPrime+=y[i]*W[i];
		}
		return kPrime;

	}


	public double [] getMatchForInstance(Instance inst) throws Exception
	{
		double c=this.classifyInstance(inst);

		double classY=0;
		int classInd=inst.classIndex();

		double [] instance=inst.toDoubleArray();
		double [] ans = new double[M+1];
		for (int i=0;i<y.length;i++)
		{
			if(W[i]==c)
				classY+=y[i];
		}
		for (int i=0;i<y.length;i++)
		{
			if(W[i]==c)
			{
				for (int j=0;j<M;j++)
				{
					if(j!=classInd)
					{
						if(instance[j]<w[j][i])
							ans[j]+=y[i]*instance[j]/classY;
						else
							ans[j]+=y[i]*w[j][i]/classY;
					}
				}
			}
			if(i>0)
				System.err.print(",");
			System.err.print(y[i]+","+W[i]);
		}
		System.err.println(","+classY+","+c);
		ans[M]=c;

		return ans;
	}





	public void writeMatlabStructure(String filename) throws Exception
	{
		FileOutputStream out; // declare a file output object
		PrintStream pts; // declare a print stream object

		// Create a new file output stream
		// connected to "myfile.txt"
		out = new FileOutputStream(filename);

		// Connect print stream to the output stream
		pts = new PrintStream( out );
		pts.print ("amap=struct(");
		pts.print ("'alpha',"+alpha);
		pts.print (",'p',"+p);
		pts.print (",'beta',"+beta);
		pts.print (",'epsilon',"+epsilon);
		pts.print (",'rho_a_bar',"+rho_a_bar);
		pts.print (",'node_threshold',"+node_threshold);
		pts.print (",'Unsupervised',0");
		pts.print (",'M',"+M);
		pts.print (",'L',"+L);

		pts.print   (",'W',[");
		for(int i=0;i<W.length;i++)
			pts.print(W[i]+" ");
		pts.print ("]");


		pts.print   (",'w',[");
		for(int i=0;i<w.length;i++)
		{
			for (int j=0;j<w[i].length;j++)
			{
				pts.print(w[i][j]+" ");
			}
			pts.print("; ");
		}
		pts.print ("]");

		pts.print(");");
		pts.close();

	}

	public double[] distributionForInstance(Instance instance) throws Exception{



		double [] classList = new double[L];

		int classlabel= (int) classifyInstance(instance);
		classList[classlabel] = 1;

		return classList;


	}

	public String toSummaryString()
	{
		return "AnalogARTMAP: 0"+",Nodes:" + y.length + ",Epsilon:"+epsilon+",Rho_a_bar:"+rho_a_bar;
	}
	/**
	 * Prints the decision tree using the private toString method from below.
	 *
	 * @return a textual description of the classifier
	 */
	public String toString() {
		String ans = "AnalogARTMAP\n\n";
		if(y!=null)
		{
			ans = ans+"Number of committed nodes: " + y.length + "\n";
			ans=ans+"Epsilon: " + epsilon + "\n";
			ans=ans+"Rho_a_bar: " + rho_a_bar + "\n";
			ans=ans+"Rho_b: " + rho_b + "\n";
			ans=ans+"Q: " + Q + "\n";
			ans=ans+"p: " + p + "\n";
			ans=ans+"Voters: " + number_of_voters + "\n";

		}
		return ans;
	}



	/**
	 Method to add a unit to F2 and F3 layers. New units Y_C and y_C get an
	 activity of 1, so be careful about that.
	 */

	public void addUnit(Instance currentInstance)
	{

		try{
			C++;
			y=expandArray(y);
			y[y.length-1]=1;
			W = expandArray(W);
			W[W.length-1]=currentInstance.classValue();


			cj = expandArray(cj);
			cj[cj.length-1] = 1;


			//LTM weights initialised too..
			w = expandArray(w,1);

			//set the weights to 1
			//for (int i=0;i<2*M;i++)
			//{
			//	w[i][C-1]=1;
			//}
//			set the weights to A
			for (int i=0;i<2*M;i++)
			{
				w[i][C-1]=A[i];
			}

		}
		catch (Exception e)
		{
			System.err.println("Exception occured in addUnit " + e.getMessage());
			e.printStackTrace();
		}
	}


	/** Takes in a one dimensional double array and adds an element at the end.
	 @param old The original one dimensional array to be copied
	 @return returns the new array
	 */
	public double [] expandArray(double [] old)
	{
		double [] n = new double[old.length+1];
		for (int i=0;i<old.length;i++)
			n[i] = old[i];
		return n;
	}





	/**
	 * Takes in a two dimensional double array and adds a row (one more
	 * index for the first index) or a column (one more index for the
	 * second index). Newly added row (column) is set to zero.
	 @param old The original array to be copied
	 @param index 0 to add a row, 1 to add a column
	 @return returns the new array
	 */

	public double [][] expandArray(double [][] old, int index)
	{


		double [][] n;
		if (index == 0)
			n = new double[old.length+1][old[0].length];
		else
			n = new double[old.length][old[0].length+1];

		for (int i=0;i<old.length;i++)
			for (int j=0; j<old[0].length;j++)
				n[i][j]=old[i][j];

		return n;
	}




	/**
	 * Main method.
	 *
	 * @param args the options for the classifier
	 */
	public static void main(String[] args) {

		try {
			AnalogARTMAP m=new AnalogARTMAP();
			System.out.println(Evaluation.evaluateModel(m, args));
			m.writeMatlabStructure(m.getClassifierFileName());
		} catch (Exception e) {
			System.err.println("Exception in main method" + e.getMessage());
			e.printStackTrace();
		}
	}




	/**
	 * Set the number of training epochs to perform.
	 * Must be greater than 0.
	 * @param n The number of epochs to train through.
	 */
	public void setTrainingTime(int n) {
		if (n > 0) {
			m_numEpochs = n;
		}
	}
	/**
	 * @return The number of epochs to train through.
	 */
	public int getTrainingTime() {
		return m_numEpochs;
	}


	/**
	 * @return the epsilon of MT
	 */
	public double getEpsilon()
	{
		return epsilon;
	}

	public void setEpsilon(double ep)
	{
		epsilon = ep;
	}

	public double getAlpha()
	{
		return alpha;
	}

	public void setAlpha(double a)
	{
		alpha=a;
	}



	public String getClassifierFileName()
	{
		return classifierFileName;
	}

	public void setClassifierFileName(String a)
	{
		classifierFileName=a;
	}

	public double getRhoBar()
	{
		return rho_a_bar;
	}


	public void setRhoBar(double r)
	{
		rho_a_bar = r;
	}

	public double getQ()
	{
		return Q;
	}

	public void setQ(double qin)
	{
		Q = qin;
	}

	public double getRhoB()
	{
		return rho_b;
	}


	public void setRhoB(double rb)
	{
		rho_b = rb;
	}

	public double getP()
	{
		return p;
	}


	public void setP(double pInp)
	{
		p = pInp;
	}

	public int getNumberOfVoters()
	{
		return number_of_voters;
	}


	public void setNumberOfVoters(int numberofvotersInp)
	{
		number_of_voters = numberofvotersInp;
	}
	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	public Enumeration listOptions() {
		Vector newVector = new Vector(1);
		newVector.addElement(new Option(
				"\tNumber of voters.\n"
				+"\t(Default = 1).",
				"V", 1,"-V <number of voters>"));
		newVector.addElement(new Option(
				"\tNumber of epochs to train through.\n"
				+"\t(Default = 1).",
				"N", 1,"-N <number of epochs>"));
		newVector.addElement(new Option(
				"\tEpsilon for MT\n"
				+"\t(Default = -0.001).",
				"e", 1,"-e <epsilon>"));

		newVector.addElement(new Option(
				"\tEpsilon for MT\n"
				+"\t(Default = 0).",
				"R", 1,"-R <baseline vigilance>"));

		newVector.addElement(new Option(
				"\tAlpha\n"
				+"\t(Default = 0.01).",
				"A", 1,"-A <alpha>"));

		newVector.addElement(new Option(
				"\tclassifierFileName\n"
				+"\t(Default = classifier.m).",
				"O", 1,"-0 <classifierfilename>"));

		newVector.addElement(new Option(
				"\tP\n"
				+"\t(Default = 2).",
				"P", 1,"-P <P>"));

		newVector.addElement(new Option(
				"\tQ\n"
				+"\t(Default = 13).",
				"Q", 1,"-Q <Q>"));

		newVector.addElement(new Option(
				"\trho_b\n"
				+"\t(Default = 0.2).",
				"B", 1,"-B <rho_b>"));

		return newVector.elements();
	}


	/**
	 * Gets the current settings of AnalogARTMAP.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String [] getOptions() {
		String [] options = new String [35];
		int current = 0;
		options[current++]="-N"; options [ current ++] = "" + getTrainingTime();
		options[current++]="-e"; options [ current ++] = "" + getEpsilon();
		options[current++]="-R"; options [ current ++] = "" + getRhoBar();
		options[current++]="-A"; options [ current ++] = "" + getAlpha();
		options[current++]="-O"; options [ current ++] = "" + getClassifierFileName();

		options[current++]="-Q"; options [ current ++] = "" + getQ();
		options[current++]="-B"; options [ current ++] = "" + getRhoB();
		options[current++]="-P"; options [ current ++] = "" + getP();
		options[current++]="-V"; options [ current ++] = "" + getNumberOfVoters();

		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

	/**
	 * Parses a given list of options. Valid options are:<p>
	 *
	 * -N num <br>
	 * Set the number of epochs to train through.
	 * (default 1) <p>
	 *
	 * @param options the list of options as an array of strings
	 * @exception Exception if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {
		//the defaults can be found here!!!!

		String numberofvotersString = Utils.getOption('V', options);
		if (numberofvotersString.length() != 0) {
			setNumberOfVoters(Integer.parseInt(numberofvotersString));
			System.err.println("Settting number of voters to " + number_of_voters);
		} else {
			setNumberOfVoters(1);
		}

		String epochsString = Utils.getOption('N', options);
		if (epochsString.length() != 0) {
			setTrainingTime(Integer.parseInt(epochsString));
			System.err.println("Settting total epochs to " + m_numEpochs);
		} else {
			setTrainingTime(1);
		}

		String epsilonString = Utils.getOption('e',options);
		if (epsilonString.length() !=0){
			setEpsilon(Double.parseDouble(epsilonString));
			System.err.println("Settting total epsilon to " + epsilon);
		}
		else
		{
			setEpsilon(-0.001);
		}

		String rhoString = Utils.getOption('R',options);
		if (rhoString.length() !=0){
			setRhoBar(Double.parseDouble(rhoString));
			System.err.println("Settting rho_a_bar to " + rho_a_bar);
		}
		else
		{
			setRhoBar(0.0);
		}

		String alphaString = Utils.getOption('A',options);
		if (alphaString.length() !=0){
			setAlpha(Double.parseDouble(alphaString));
			System.err.println("Settting alpha to " + alpha);
		}
		else
		{
			setAlpha(0.01);
		}


		String classifierName= Utils.getOption('O',options);
		if (alphaString.length() !=0){
			System.err.println("Settting alpha to " + classifierName);
		}
		else
		{
			setClassifierFileName(classifierName);
		}

		String qString = Utils.getOption('Q',options);
		if (qString.length() !=0){
			setQ(Double.parseDouble(qString));
			System.err.println("Settting Q to " + Q);
		}
		else
		{
			setQ(13);
		}

		String pString = Utils.getOption('P',options);
		if (pString.length() !=0){
			setP(Double.parseDouble(pString));
			System.err.println("Settting P to " + p);
		}
		else
		{
			setP(2);
		}

		String rhobString = Utils.getOption('B',options);
		if (rhobString.length() !=0){
			setRhoB(Double.parseDouble(rhobString));
			System.err.println("Settting Rho B to " + rho_b);
		}
		else
		{
			setRhoB(0.2);
		}
		//

		Utils.checkForRemainingOptions(options);
	}
}

