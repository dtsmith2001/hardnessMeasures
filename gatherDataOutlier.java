/**
 * Michael R. Smith 2010
 */

import java.util.Random;


import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddID;

public class gatherDataOutlier
{
	private enum Learner
	{
		CL_C4dot5, C4dot5, NB, BackProp, Perceptron, SVM, IB1, IB5, RIPPER, RBFNet, BackPropLarge, PerceptronLarge, AdaBoostMLP, MultiBoostMLP, BaggingMLP, AdaBoost, MultiBoost, AdaBoostNB, MultiBoostNB, Ridor, NNge, LWL, RandForest, BackPropConverge
	}
	
	private static Learner learner;
	
	public static String[] getOptions(String ignore) throws Exception
	{
		int filterSize = 3; //The number of options for the filter (this ignores the first attribute which is the ID)
		String[] classifier = new String[1];
		
	  	switch (learner)
   	{
	  	case CL_C4dot5:
	  		classifier = new String[13 + filterSize];
	   	classifier[0 + filterSize] = "hardnessTraining.classifiers.curriculumLearning.J48";
	   	classifier[1 + filterSize] = "--";
	   	classifier[2 + filterSize] = "-hardVals";
	   	classifier[3 + filterSize] = "/home/msmith/Research/instanceComplexity/instanceHardness/autos.out";
	   	classifier[4 + filterSize] = "-ihThresh";
	   	classifier[5 + filterSize] = "1.4";
	   	classifier[6 + filterSize] = "-ihMax";
	   	classifier[7 + filterSize] = "0.75";
	   	classifier[8 + filterSize] = "-IP";
	   	classifier[9 + filterSize] = "-C";
	   	classifier[10 + filterSize] = "0.25";
	   	classifier[11 + filterSize] = "-M";
	   	classifier[12 + filterSize] = "2";
	   	break; 
	  	case C4dot5:
	  		classifier = new String[6 + filterSize];
	   	classifier[0 + filterSize] = "weka.classifiers.trees.J48";
	   	classifier[1 + filterSize] = "--";
	   	classifier[2 + filterSize] = "-C";
	   	classifier[3 + filterSize] = "0.25";
	   	classifier[4 + filterSize] = "-M";
	   	classifier[5 + filterSize] = "2";
	   	break;
	   
	  	case NB:
	  		classifier = new String[2 + filterSize];
	   	classifier[0 + filterSize] = "weka.classifiers.bayes.NaiveBayes";
	   	classifier[1 + filterSize] = "--";
	   	break;
	   	
	  	case BackProp:
	  		classifier = new String[16 + filterSize];
	   	classifier[0 + filterSize] = "weka.classifiers.functions.MultilayerPerceptron";
	  		//classifier[0 + filterSize] = "hardnessTraining.classifiers.curriculumLearning.MultilayerPerceptron_Curric";
	   	classifier[1 + filterSize] = "--";
	   	classifier[2 + filterSize] = "-L";
	   	classifier[3 + filterSize] = "0.3";
	   	classifier[4 + filterSize] = "-M";
	   	classifier[5 + filterSize] = "0.2";
	   	classifier[6 + filterSize] = "-N";
	   	classifier[7 + filterSize] = "500";
	   	classifier[8 + filterSize] = "-V";
	   	classifier[9 + filterSize] = "0";
	   	classifier[10 + filterSize] = "-S";
	   	classifier[11 + filterSize] = "0";
	   	classifier[12 + filterSize] = "-E";
	   	classifier[13 + filterSize] = "20";
	   	classifier[14 + filterSize] = "-H";
	   	classifier[15 + filterSize] = "a";
	   	//This is to not reset the value.  Trying it for the datasets that take a long time to run
	   	//classifier[16 + filterSize] = "-R";
	   	break;
	   	
	  	case BackPropConverge:
	  		classifier = new String[16 + filterSize];
	   	//classifier[0 + filterSize] = "weka.classifiers.functions.MultilayerPerceptron";
	  		classifier[0 + filterSize] = "hardnessTraining.classifiers.curriculumLearning.MultilayerPerceptron_Curric";
	   	classifier[1 + filterSize] = "--";
	   	classifier[2 + filterSize] = "-L";
	   	classifier[3 + filterSize] = "0.3";
	   	classifier[4 + filterSize] = "-M";
	   	classifier[5 + filterSize] = "0.2";
	   	classifier[6 + filterSize] = "-N";
	   	classifier[7 + filterSize] = "500";
	   	classifier[8 + filterSize] = "-V";
	   	classifier[9 + filterSize] = "0";
	   	classifier[10 + filterSize] = "-S";
	   	classifier[11 + filterSize] = "0";
	   	classifier[12 + filterSize] = "-E";
	   	classifier[13 + filterSize] = "20";
	   	classifier[14 + filterSize] = "-H";
	   	classifier[15 + filterSize] = "a";
	   	//This is to not reset the value.  Trying it for the datasets that take a long time to run
	   	//classifier[16 + filterSize] = "-R";
	   	break;
	   	
	  	case BackPropLarge:
	  		classifier = new String[17 + filterSize];
	   	classifier[0 + filterSize] = "weka.classifiers.functions.MultilayerPerceptron";
	   	classifier[1 + filterSize] = "--";
	   	classifier[2 + filterSize] = "-L";
	   	classifier[3 + filterSize] = "0.3";
	   	classifier[4 + filterSize] = "-M";
	   	classifier[5 + filterSize] = "0.2";
	   	classifier[6 + filterSize] = "-N";
	   	classifier[7 + filterSize] = "500";
	   	classifier[8 + filterSize] = "-V";
	   	classifier[9 + filterSize] = "15";
	   	classifier[10 + filterSize] = "-S";
	   	classifier[11 + filterSize] = "0";
	   	classifier[12 + filterSize] = "-E";
	   	classifier[13 + filterSize] = "20";
	   	classifier[14 + filterSize] = "-H";
	   	classifier[15 + filterSize] = "a";
	   	classifier[16 + filterSize] = "-B";
	   	//classifier[17 + filterSize] = "-R";
	   	break;
	   	
	   	//Use an MLP with 0 hidden layers
	   	//Voted Perceptron
	  	case Perceptron:
	  		classifier = new String[16 + filterSize];
	   	classifier[0 + filterSize] = "weka.classifiers.functions.MultilayerPerceptron";
	   	classifier[1 + filterSize] = "--";
	   	classifier[2 + filterSize] = "-L";
	   	classifier[3 + filterSize] = "0.3";
	   	classifier[4 + filterSize] = "-M";
	   	classifier[5 + filterSize] = "0.2";
	   	classifier[6 + filterSize] = "-N";
	   	classifier[7 + filterSize] = "500";
	   	classifier[8 + filterSize] = "-V";
	   	classifier[9 + filterSize] = "0";
	   	classifier[10 + filterSize] = "-S";
	   	classifier[11 + filterSize] = "0";
	   	classifier[12 + filterSize] = "-E";
	   	classifier[13 + filterSize] = "20";
	   	classifier[14 + filterSize] = "-H";
	   	classifier[15 + filterSize] = "0";
	   	break;
	   	
	  	case PerceptronLarge:
	  		classifier = new String[17 + filterSize];
	   	classifier[0 + filterSize] = "weka.classifiers.functions.MultilayerPerceptron";
	   	classifier[1 + filterSize] = "--";
	   	classifier[2 + filterSize] = "-L";
	   	classifier[3 + filterSize] = "0.3";
	   	classifier[4 + filterSize] = "-M";
	   	classifier[5 + filterSize] = "0.2";
	   	classifier[6 + filterSize] = "-N";
	   	classifier[7 + filterSize] = "500";
	   	classifier[8 + filterSize] = "-V";
	   	classifier[9 + filterSize] = "0";
	   	classifier[10 + filterSize] = "-S";
	   	classifier[11 + filterSize] = "0";
	   	classifier[12 + filterSize] = "-E";
	   	classifier[13 + filterSize] = "20";
	   	classifier[14 + filterSize] = "-H";
	   	classifier[15 + filterSize] = "0";
	   	classifier[16 + filterSize] = "-B";
	   	break;
	  			
	  	case SVM:
	  		classifier = new String[16 + filterSize];
	   	classifier[0 + filterSize] = "weka.classifiers.functions.SMO";
	   	classifier[1 + filterSize] = "--";
	   	classifier[2 + filterSize] = "-C";
	   	classifier[3 + filterSize] = "1.0";
	   	classifier[4 + filterSize] = "-L";
	   	classifier[5 + filterSize] = "0.0010";
	   	classifier[6 + filterSize] = "-P";
	   	classifier[7 + filterSize] = "1.0E-12";
	   	classifier[8 + filterSize] = "-N";
	   	classifier[9 + filterSize] = "0";
	   	classifier[10 + filterSize] = "-V";
	   	classifier[11 + filterSize] = "-1";
	   	classifier[12 + filterSize] = "-W";
	   	classifier[13 + filterSize] = "1";
	   	classifier[14 + filterSize] = "-K";
	   	classifier[15 + filterSize] = "weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 2.0";
	   	break;
	   	
	  	case IB1:
	  		classifier = new String[1 + filterSize];
	   	classifier[0 + filterSize] = "weka.classifiers.lazy.IB1";
	   	break;
	   	
   	case IB5:
   		classifier = new String[7 + filterSize];
   		classifier[0 + filterSize] = "weka.classifiers.lazy.IBk";
   		classifier[1 + filterSize] = "-K";
   		classifier[2 + filterSize] = "5";
   		classifier[3 + filterSize] = "-W";
   		classifier[4 + filterSize] = "0";
   		classifier[5 + filterSize] = "-A";
   		classifier[6 + filterSize] = "weka.core.neighboursearch.LinearNNSearch - A \"weka.core.EuclideanDistance -R first-last\"";
   		break;
   		
   	case RIPPER:
   		classifier = new String[10 + filterSize];
      	classifier[0 + filterSize] = "weka.classifiers.rules.JRip";
      	classifier[1 + filterSize] = "--";
      	classifier[2 + filterSize] = "-F";
      	classifier[3 + filterSize] = "3";
      	classifier[4 + filterSize] = "-N";
      	classifier[5 + filterSize] = "2.0";
      	classifier[6 + filterSize] = "-O";
      	classifier[7 + filterSize] = "2";
      	classifier[8 + filterSize] = "-S";
      	classifier[9 + filterSize] = "1";
      	break;
      	
   	case RBFNet:
   		classifier = new String[12 + filterSize];
      	classifier[0 + filterSize] = "weka.classifiers.functions.RBFNetwork";
      	classifier[1 + filterSize] = "--";
      	classifier[2 + filterSize] = "-B";
      	classifier[3 + filterSize] = "2";
      	classifier[4 + filterSize] = "-S";
      	classifier[5 + filterSize] = "1";
      	classifier[6 + filterSize] = "-R";
      	classifier[7 + filterSize] = "1.0E-8";
      	classifier[8 + filterSize] = "-M";
      	classifier[9 + filterSize] = "10000";
      	classifier[10 + filterSize] = "-W";
      	classifier[11 + filterSize] = "0.1";
      	//This is what it should be,  I changed it for testing
      	//classifier[9 + filterSize] = "10000";
      	break;
      	
   	case AdaBoostMLP:
   		classifier = new String[25 + filterSize];
      	classifier[0 + filterSize] = "weka.classifiers.meta.AdaBoostM1";
      	classifier[1 + filterSize] = "--";
			classifier[2 + filterSize] = "-P";
			classifier[3 + filterSize] = "100";
			classifier[4 + filterSize] = "-S";
			classifier[5 + filterSize] = "1";
			classifier[6 + filterSize] = "-I";
			classifier[7 + filterSize] = "10";
			classifier[8 + filterSize] = "-W";
			classifier[9 + filterSize] = "weka.classifiers.functions.MultilayerPerceptron";
			classifier[10 + filterSize] = "--";
			classifier[11 + filterSize] = "-L";
			classifier[12 + filterSize] = "0.3";
			classifier[13 + filterSize] = "-M"; 
			classifier[14 + filterSize] = "0.2";
			classifier[15 + filterSize] = "-N";
			classifier[16 + filterSize] = "500";
			classifier[17 + filterSize] = "-V"; 
			classifier[18 + filterSize] = "0"; 
			classifier[19 + filterSize] = "-S";
			classifier[20 + filterSize] = "0";
			classifier[21 + filterSize] = "-E";
			classifier[22 + filterSize] = "20";
			classifier[23 + filterSize] = "-H";
			classifier[24 + filterSize] = "a";
      	//This is what it should be,  I changed it for testing
      	//classifier[9 + filterSize] = "10000";
      	break;
      	
   	case MultiBoostMLP:
   		classifier = new String[27 + filterSize];
      	classifier[0 + filterSize] = "weka.classifiers.meta.MultiBoostAB";
      	classifier[1 + filterSize] = "--";
			classifier[2 + filterSize] = "-C";
			classifier[3 + filterSize] = "3";
			classifier[4 + filterSize] = "-P";
			classifier[5 + filterSize] = "100";
			classifier[6 + filterSize] = "-S";
			classifier[7 + filterSize] = "1";
			classifier[8 + filterSize] = "-I";
			classifier[9 + filterSize] = "10";
			classifier[10 + filterSize] = "-W";
			classifier[11 + filterSize] = "weka.classifiers.functions.MultilayerPerceptron";
			classifier[12 + filterSize] = "--";
			classifier[13 + filterSize] = "-L";
			classifier[14 + filterSize] = "0.3";
			classifier[15 + filterSize] = "-M"; 
			classifier[16 + filterSize] = "0.2";
			classifier[17 + filterSize] = "-N";
			classifier[18 + filterSize] = "500";
			classifier[19 + filterSize] = "-V"; 
			classifier[20 + filterSize] = "0"; 
			classifier[21 + filterSize] = "-S";
			classifier[22 + filterSize] = "0";
			classifier[23 + filterSize] = "-E";
			classifier[24 + filterSize] = "20";
			classifier[25 + filterSize] = "-H";
			classifier[26 + filterSize] = "a";
      	//This is what it should be,  I changed it for testing
      	//classifier[9 + filterSize] = "10000";
      	break;
      	
   	case AdaBoost:
   		classifier = new String[8 + filterSize];
      	classifier[0 + filterSize] = "weka.classifiers.meta.AdaBoostM1";
      	classifier[1 + filterSize] = "--";
			classifier[2 + filterSize] = "-P";
			classifier[3 + filterSize] = "100";
			classifier[4 + filterSize] = "-S";
			classifier[5 + filterSize] = "1";
			classifier[6 + filterSize] = "-I";
			classifier[7 + filterSize] = "10";
      	break;
      	
   	case MultiBoost:
   		classifier = new String[10 + filterSize];
      	classifier[0 + filterSize] = "weka.classifiers.meta.MultiBoostAB";
      	classifier[1 + filterSize] = "--";
			classifier[2 + filterSize] = "-C";
			classifier[3 + filterSize] = "3";
			classifier[4 + filterSize] = "-P";
			classifier[5 + filterSize] = "100";
			classifier[6 + filterSize] = "-S";
			classifier[7 + filterSize] = "1";
			classifier[8 + filterSize] = "-I";
			classifier[9 + filterSize] = "10";
      	break;
      	
   	case BaggingMLP:
   		classifier = new String[25 + filterSize];
      	classifier[0 + filterSize] = "weka.classifiers.meta.Bagging";
      	classifier[1 + filterSize] = "--";
			classifier[2 + filterSize] = "-P";
			classifier[3 + filterSize] = "100";
			classifier[4 + filterSize] = "-S";
			classifier[5 + filterSize] = "1";
			classifier[6 + filterSize] = "-I";
			classifier[7 + filterSize] = "10";
			classifier[8 + filterSize] = "-W";
			classifier[9 + filterSize] = "weka.classifiers.functions.MultilayerPerceptron";
			classifier[10 + filterSize] = "--";
			classifier[11 + filterSize] = "-L";
			classifier[12 + filterSize] = "0.3";
			classifier[13 + filterSize] = "-M"; 
			classifier[14 + filterSize] = "0.2";
			classifier[15 + filterSize] = "-N";
			classifier[16 + filterSize] = "500";
			classifier[17 + filterSize] = "-V"; 
			classifier[18 + filterSize] = "0"; 
			classifier[19 + filterSize] = "-S";
			classifier[20 + filterSize] = "0";
			classifier[21 + filterSize] = "-E";
			classifier[22 + filterSize] = "20";
			classifier[23 + filterSize] = "-H";
			classifier[24 + filterSize] = "a";
      	//This is what it should be,  I changed it for testing
      	//classifier[9 + filterSize] = "10000";
      	break;
      	
   	case AdaBoostNB:
   		classifier = new String[11 + filterSize];
      	classifier[0 + filterSize] = "weka.classifiers.meta.AdaBoostM1";
      	classifier[1 + filterSize] = "--";
			classifier[2 + filterSize] = "-P";
			classifier[3 + filterSize] = "100";
			classifier[4 + filterSize] = "-S";
			classifier[5 + filterSize] = "1";
			classifier[6 + filterSize] = "-I";
			classifier[7 + filterSize] = "10";
			classifier[8 + filterSize] = "-W";
			classifier[9 + filterSize] = "weka.classifiers.bayes.NaiveBayes";
			classifier[10 + filterSize] = "--";
      	//This is what it should be,  I changed it for testing
      	//classifier[9 + filterSize] = "10000";
      	break;
      	
   	case MultiBoostNB:
   		classifier = new String[13 + filterSize];
      	classifier[0 + filterSize] = "weka.classifiers.meta.MultiBoostAB";
      	classifier[1 + filterSize] = "--";
			classifier[2 + filterSize] = "-C";
			classifier[3 + filterSize] = "3";
			classifier[4 + filterSize] = "-P";
			classifier[5 + filterSize] = "100";
			classifier[6 + filterSize] = "-S";
			classifier[7 + filterSize] = "1";
			classifier[8 + filterSize] = "-I";
			classifier[9 + filterSize] = "10";
			classifier[10 + filterSize] = "-W";
			classifier[11 + filterSize] = "weka.classifiers.bayes.NaiveBayes";
			classifier[12 + filterSize] = "--";
      	//This is what it should be,  I changed it for testing
      	//classifier[9 + filterSize] = "10000";
      	break;
      	
   	case Ridor:
   		classifier = new String[8 + filterSize];
      	classifier[0 + filterSize] = "weka.classifiers.rules.Ridor";
      	classifier[1 + filterSize] = "--";
      	classifier[2 + filterSize] = "-F";
      	classifier[3 + filterSize] = "3";
      	classifier[4 + filterSize] = "-S";
      	classifier[5 + filterSize] = "1";
      	classifier[6 + filterSize] = "-N";
      	classifier[7 + filterSize] = "2.0";
   		break;
   		
   	case NNge:
   		classifier = new String[6 + filterSize];
      	classifier[0 + filterSize] = "weka.classifiers.rules.NNge";
      	classifier[1 + filterSize] = "--";
      	classifier[2 + filterSize] = "-G";
      	classifier[3 + filterSize] = "5";
      	classifier[4 + filterSize] = "-I";
      	classifier[5 + filterSize] = "5";
      	break;
      	
   	case LWL:
   		classifier = new String[10 + filterSize];
      	classifier[0 + filterSize] = "weka.classifiers.lazy.LWL";
      	classifier[1 + filterSize] = "--";
      	classifier[2 + filterSize] = "-U";
      	classifier[3 + filterSize] = "0";
      	classifier[4 + filterSize] = "-K";
      	classifier[5 + filterSize] = "-1";
      	classifier[6 + filterSize] = "-A";
      	classifier[7 + filterSize] = "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"";
      	classifier[8 + filterSize] = "-W";
      	classifier[9 + filterSize] = "weka.classifiers.trees.DecisionStump";
   		break;
   		
   	case RandForest:
   		classifier = new String[8 + filterSize];
      	classifier[0 + filterSize] = "weka.classifiers.trees.RandomForest";
      	classifier[1 + filterSize] = "--";
      	classifier[2 + filterSize] = "-I";
      	classifier[3 + filterSize] = "10";
      	classifier[4 + filterSize] = "-K";
      	classifier[5 + filterSize] = "0";
      	classifier[6 + filterSize] = "-S";
      	classifier[7 + filterSize] = "1";
   		break;
      	
   	default:
   		throw new Exception("Not a valid Learner choice");
   	}

	  	classifier[0] = "-F";
	  	classifier[1] = "weka.filters.unsupervised.attribute.Remove -R " + ignore;
	  	classifier[2] = "-W";
		return classifier;
	}
	
	public static void main(String[] args) throws Exception
   {
		String strLearner;	//Name of the learning algorithm to use
		String dataset;		//Name of the dataset to use
      DataSource source;	//Source of the dataset
      Instances data;		//Original data
      int classIndex;		//Index of the class attribute
      MyEvaluation eval;		//Object to evaluate the classifier

		//For testing purposes
		//System.out.println("Testing Here Dude" + args.length);
		
		//Get the learning algorithm and dataset name 
      //Make sure there are at least two arguments
   	if (args.length < 2)
   	{
   		throw new Exception("Usage: gatherData {Learner} {Dataset} {RandomSeed} [-c classIndex]\n");
   	}
		strLearner = args[0];
		dataset = args[1];
		
		//Initialize the dataset
		source = new DataSource(dataset);
		data = source.getDataSet();
		
		//Add an ID attribute to keep track of the instances
      if (data.attribute(1).name().toLowerCase() != "id" || data.attribute(1).isNumeric())
      {
        	//System.out.println(data.numAttributes());
     	   String[] IDOptions = new String[4];
     	   IDOptions[0] = "-i";
        	IDOptions[1] = dataset;
        	IDOptions[2] = "-o";
        	IDOptions[3] = dataset.replace(".arff", "-id.arff");
        	//Add the id attribute
     	   AddID addId = new AddID();
        	addId.setOptions(IDOptions);
        	addId.setInputFormat(data);
       	data = Filter.useFilter(data, addId);
      }
      
      
      //Set the index of the class attribute
		String classIdx = Utils.getOption('c',args);
      if (classIdx.length() != 0) 
      {
        classIndex = Integer.parseInt(classIdx);
        if ((classIndex <= 0) || (classIndex - 1 >= data.numAttributes())) throw new Exception("Invalid value for class index!");
      } 
      else 
      {
        classIndex = data.numAttributes() - 1;
      }
      data.setClassIndex(classIndex);
		
      //Which attributes to ignore use base 1 since 1 attribute will be added
		String ignore = Utils.getOption('i',args);
      ignore = "1,2," + ignore;

		//Create the Classifier
      Classifier classify = new FilteredClassifier();
      
      //Get the options based on the chosen learning algorithm
      learner = Enum.valueOf(Learner.class, strLearner);
      String[] temp = getOptions(ignore);
   	String[] options = new String[temp.length + args.length - 3];
   	
   	for (int i = 0; i < temp.length; i++)
   		options[i] = temp[i];
      
      for (int i=3; i < args.length; i++)
      	options[i + temp.length - 3] = args[i];
   	
   	
   	classify.setOptions(options);
   	
		//Evaluate the data using 10-fold cross validation
   	eval = new MyEvaluation(data);
   	
		//Write output to a file:
		//Name file: {Dataset}.{Learner}.dat
		// Instance# Actual Predicted
   	//
   	eval.toSummaryStringCrossValidateModelOutlier(classify, data, 10, new Random(Integer.parseInt(args[2])));
   	System.out.println(eval.toSummaryString());
   }
}
