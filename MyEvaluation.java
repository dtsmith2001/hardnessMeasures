/**
 * Michael R. Smith 2011
 */

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.HashMap;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
//import weka.core.FastVector;
//import weka.core.Instance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Range;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.AddID;

public class MyEvaluation extends weka.classifiers.Evaluation 
{
   protected HashMap<Integer,Integer> wrongPredictions;
   
   /**
    * Constructor
    * @param data
    * @throws Exception
    */
   public MyEvaluation(Instances data) throws Exception
   {
	  super(data);
     wrongPredictions = new HashMap<Integer, Integer>();
   }
   
   /**
    * Returns the hash map of wrong predictions.
    * @return
    */
   public HashMap<Integer, Integer> getWrongPredictions()
   {
   	return wrongPredictions;
   }
   
   public void toSummaryStringCrossValidateModelOutlier(Classifier classifier, Instances data, 
         int numFolds, Random random) 
   throws Exception 
   {
   	data = new Instances(data);
   	//Instances data2 = new Instances(data);
   	Instances outliers = new Instances(data, 0);
   	for (int i = 0; i < data.numInstances(); i++)
   	{
   		//System.out.println(data.instance(i).value(1) == 0);
   		if (data.instance(i).value(1)==0)
   		{
   			System.out.println(data.instance(i).value(0));
   			outliers.add(data.instance(i));
      		data.delete(i);
      		i--;
   		}
   	}
   	if (data.numInstances() < numFolds)
   	{
   		numFolds=data.numInstances();
   	}
	  // Make a copy of the data we can reorder
   	//System.out.println(data.numAttributes());
	  
	  data.randomize(random);
	  //data2.randomize(random);
	  double pred;
	  if (data.classAttribute().isNominal()) 
	  {
         data.stratify(numFolds);
	  }

	  /*if (data2.classAttribute().isNominal()) 
	  {
         data2.stratify(numFolds);
	  }*/
	  
	  System.out.print("Inst#\tActual\tPredicted\tfold#\n");
	  // Do the folds
	  for (int i = 0; i < numFolds; i++) 
	  {
        Instances train = data.trainCV(numFolds, i, random);
	     setPriors(train);
/*	     for (int j = 0; j < train.numInstances(); j++)
	     {
	   	  System.out.println(train.instance(j).value(0));
	     }*/
	     Classifier copiedClassifier = Classifier.makeCopy(classifier);
	     copiedClassifier.buildClassifier(train);
	     Instances test = data.testCV(numFolds, i);
        //System.out.println(test.instance(0).toString(0));
	     //evaluateModel(copiedClassifier, test);
	     for (int j = 0; j < test.numInstances(); j++)
	     {
	   	  //write to file here
	   	  if (data.classAttribute().isNumeric())
	   	  {
		   	  pred = evaluateModelOnce(copiedClassifier, test.instance(j));
	   		  System.out.print(test.instance(j).value(0) + "\t" + test.instance(j).classValue() + "\t" + pred + "\t" + i + "\n");
	   	  }
	   	  else
	   	  {
		   	  pred = evaluateModelOnce(copiedClassifier, test.instance(j));
	   		  System.out.print(test.instance(j).value(0) + "\t" + data.classAttribute().value((int)test.instance(j).classValue()) + "\t" + data.classAttribute().value((int)pred) + "\t" + i + "\n");
	   	  }
	   	  //pred = (int)evaluateModelOnce(copiedClassifier, test.instance(j));
	   	  //System.out.print(test.instance(j).value(0) + "\t" + data.classAttribute().value((int)test.instance(j).classValue()) + "\t" + data.classAttribute().value(pred) + "\t" + i + "\n");
	   	  /*if (pred != test.instance(j).classValue())
	   	  {
	   		  pred = (int)test.instance(j).value(0);
	   		  if (!wrongPredictions.containsKey(pred))
	   		  {
	   			  wrongPredictions.put(pred, 1);
	   		  }
	   		  else
	   		  {
	   			  wrongPredictions.put(pred, wrongPredictions.get(pred) + 1);
	   		  }
	   		  
	   	  }*/
	   	  
	     }
	     
	  }
	  Classifier newClassifier = Classifier.makeCopy(classifier);
	  newClassifier.buildClassifier(data);
	  for (int i = 0; i < outliers.numInstances(); i++)
	  {

   	  if (data.classAttribute().isNumeric())
   	  {
	   	  pred = evaluateModelOnce(newClassifier, outliers.instance(i));
   		  System.out.print(outliers.instance(i).value(0) + "\t" + outliers.instance(i).classValue() + "\t" + pred + "\tlast" + "\n");
   	  }
   	  else
   	  {
	   	  pred = evaluateModelOnce(newClassifier, outliers.instance(i));
	   	  System.out.print(outliers.instance(i).value(0) + "\t" + data.classAttribute().value((int)outliers.instance(i).classValue()) + "\t" + data.classAttribute().value((int)pred) + "\tlast" + "\n");
   	  }
		  //pred = evaluateModelOnce(newClassifier, outliers.instance(i));
		  //System.out.print(outliers.instance(i).value(0) + "\t" + data.classAttribute().value((int)outliers.instance(i).classValue()) + "\t" + data.classAttribute().value((int)pred) + "\tlast" + "\n");
	  }
      m_NumFolds = numFolds;
   }
   
   public void toSummaryStringCrossValidateModelSMOTE(Classifier classifier, Instances data, 
         int numFolds, Random random) 
   throws Exception 
   {
   	data = new Instances(data);
   	//Instances data2 = new Instances(data);
   	
   	if (data.numInstances() < numFolds)
   	{
   		numFolds=data.numInstances();
   	}
	  // Make a copy of the data we can reorder
   	//System.out.println(data.numAttributes());
	  
	  data.randomize(random);
	  int pred;
	  if (data.classAttribute().isNominal()) 
	  {
         data.stratify(numFolds);
	  }
	  
	  System.out.print("Inst#\tActual\tPredicted\tfold#\n");
	  // Do the folds
	  for (int i = 0; i < numFolds; i++) 
	  {
        Instances train = data.trainCV(numFolds, i, random);
     	  Instances[] classes = new Instances[data.numClasses()];
        
        //System.out.println(train.numInstances());
        
        for (int j = 0; j < classes.length; j++)
        {
           classes[j] = new Instances(data,0);
        }
        for (int j = 0; j < train.numInstances(); j++)
        {
           classes[(int)train.instance(j).value(train.classIndex())].add(train.instance(j));
        }

        /*for (int j = 0; j < classes.length; j++)
        {
           System.out.println(j + " " + classes[j].numInstances());
        }*/
        
        int max = 0;
        int maxClass = 0;
        for (int j = 0; j < classes.length; j++)
        {
           if (classes[j].numInstances() > max)
       	  {
       	     max = classes[j].numInstances();
       		  maxClass = j;
       	  }
        }
        //System.out.println(max);
        
       	for (int k = 0; k < classes.length; k++)
       	{
       		if (k != maxClass && classes[k].numInstances() > 1 && classes[k].numInstances() != max)
       		{
       			
       	   String[] IDOptions = new String[8];
        	   IDOptions[0] = "-C";
            IDOptions[1] = Integer.toString(k + 1);
            IDOptions[2] = "-K";
            if (classes[k].numInstances() > 5)
            {
               IDOptions[3] = "5";
            }
            else
            {
            	IDOptions[3] = Integer.toString(classes[k].numInstances() - 1);
            }
            
            IDOptions[4] = "-P";
            //IDOptions[5] = Double.toString((1 - (classes[k].numInstances() * 1.0/max)) * 100);
            IDOptions[5] = Double.toString(((max * 1.0 /classes[k].numInstances())-1) * 100);
            IDOptions[6] = "-S";
            IDOptions[7] = "1";
            
            //System.out.println("DUDE\t" + IDOptions[5] + " " + classes[k].numInstances() + "/" + max + " = " + classes[k].numInstances() * 1.0/max);
            
           	//Add the id attribute
        	   SMOTE smote = new SMOTE();
           	smote.setOptions(IDOptions);
           	smote.setInputFormat(data);
          	train = Filter.useFilter(train, smote);
          	
          	
       		}
       		
       	}

    		/*ArffSaver saver = new ArffSaver();
    		saver.setInstances(train);
       	saver.setFile(new File("/home/msmith/Desktop/cotact-Leneses" + Integer.toString(i) + ".arff"));
       	saver.writeBatch();*/
        
       	//System.out.println("HERE -> " + train.numInstances());


         //if (true)
        	 //return;
	     setPriors(train);

	     Classifier copiedClassifier = Classifier.makeCopy(classifier);
	     copiedClassifier.buildClassifier(train);
	     Instances test = data.testCV(numFolds, i);
        //System.out.println(test.instance(0).toString(0));
	     //evaluateModel(copiedClassifier, test);
	     for (int j = 0; j < test.numInstances(); j++)
	     {
	   	  //write to file here
	   	  pred = (int)evaluateModelOnce(copiedClassifier, test.instance(j));
	   	  System.out.print(test.instance(j).value(0) + "\t" + data.classAttribute().value((int)test.instance(j).classValue()) + "\t" + data.classAttribute().value(pred) + "\t" + i + "\n");
	   	  /*if (pred != test.instance(j).classValue())
	   	  {
	   		  pred = (int)test.instance(j).value(0);
	   		  if (!wrongPredictions.containsKey(pred))
	   		  {
	   			  wrongPredictions.put(pred, 1);
	   		  }
	   		  else
	   		  {
	   			  wrongPredictions.put(pred, wrongPredictions.get(pred) + 1);
	   		  }
	   		  
	   	  }*/
	   	  
	     }
	     
	  }
      m_NumFolds = numFolds;
   }
   
   public void toSummaryStringCrossValidateModelOutlierSMOTE(Classifier classifier, Instances data, 
         int numFolds, Random random) 
   throws Exception 
   {
   	data = new Instances(data);
   	//Instances data2 = new Instances(data);
   	Instances outliers = new Instances(data, 0);
   	for (int i = 0; i < data.numInstances(); i++)
   	{
   		//System.out.println(data.instance(i).value(1) == 0);
   		if (data.instance(i).value(1)==0)
   		{
   			System.out.println(data.instance(i).value(0));
   			outliers.add(data.instance(i));
      		data.delete(i);
      		i--;
   		}
   	}
   	if (data.numInstances() < numFolds)
   	{
   		numFolds=data.numInstances();
   	}
	  // Make a copy of the data we can reorder
   	//System.out.println(data.numAttributes());
	  
	  data.randomize(random);
	  //data2.randomize(random);
	  int pred;
	  if (data.classAttribute().isNominal()) 
	  {
         data.stratify(numFolds);
	  }

	  /*if (data2.classAttribute().isNominal()) 
	  {
         data2.stratify(numFolds);
	  }*/
	  
	  System.out.print("Inst#\tActual\tPredicted\tfold#\n");
	  // Do the folds
	  for (int i = 0; i < numFolds; i++) 
	  {
        Instances train = data.trainCV(numFolds, i, random);
     	  Instances[] classes = new Instances[data.numClasses()];
        
        
        
        for (int j = 0; j < classes.length; j++)
        {
           classes[j] = new Instances(data,0);
        }
        for (int j = 0; j < train.numInstances(); j++)
        {
           classes[(int)train.instance(j).value(train.classIndex())].add(train.instance(j));
        }
          
        int max = 0;
        int maxClass = 0;
        for (int j = 0; j < classes.length; j++)
        {
           if (classes[j].numInstances() > max)
       	  {
       	     max = classes[j].numInstances();
       		  maxClass = j;
       	  }
        }
        //System.out.println(max + " " + maxClass);
       	//if (true)
//       		return;
       	
       	for (int k = 0; k < classes.length; k++)
       	{
       		if (k != maxClass && classes[k].numInstances() > 1 && classes[k].numInstances() != max)
       		{
       			
       	   String[] IDOptions = new String[8];
        	   IDOptions[0] = "-C";
            IDOptions[1] = Integer.toString(k + 1);
            IDOptions[2] = "-K";
            if (classes[k].numInstances() > 5)
            {
               IDOptions[3] = "5";
            }
            else
            {
            	IDOptions[3] = Integer.toString(classes[k].numInstances() - 1);
            }
            
            IDOptions[4] = "-P";
            //IDOptions[5] = Double.toString((1 - (classes[k].numInstances()/max)) * 100);
            IDOptions[5] = Double.toString(((max * 1.0 /classes[k].numInstances())-1) * 100);
            IDOptions[6] = "-S";
            IDOptions[7] = "1";
           	//Add the id attribute
        	   SMOTE addId = new SMOTE();
           	addId.setOptions(IDOptions);
           	addId.setInputFormat(train);
          	train = Filter.useFilter(train, addId);
       		}
       		
       	}
        
        
        
	     setPriors(train);
/*	     for (int j = 0; j < train.numInstances(); j++)
	     {
	   	  System.out.println(train.instance(j).value(0));
	     }*/
	     Classifier copiedClassifier = Classifier.makeCopy(classifier);
	     copiedClassifier.buildClassifier(train);
	     Instances test = data.testCV(numFolds, i);
        //System.out.println(test.instance(0).toString(0));
	     //evaluateModel(copiedClassifier, test);
	     for (int j = 0; j < test.numInstances(); j++)
	     {
	   	  //write to file here
	   	  pred = (int)evaluateModelOnce(copiedClassifier, test.instance(j));
	   	  System.out.print(test.instance(j).value(0) + "\t" + data.classAttribute().value((int)test.instance(j).classValue()) + "\t" + data.classAttribute().value(pred) + "\t" + i + "\n");
	   	  /*if (pred != test.instance(j).classValue())
	   	  {
	   		  pred = (int)test.instance(j).value(0);
	   		  if (!wrongPredictions.containsKey(pred))
	   		  {
	   			  wrongPredictions.put(pred, 1);
	   		  }
	   		  else
	   		  {
	   			  wrongPredictions.put(pred, wrongPredictions.get(pred) + 1);
	   		  }
	   		  
	   	  }*/
	   	  
	     }
	     
	  }
	  Classifier newClassifier = Classifier.makeCopy(classifier);
	  newClassifier.buildClassifier(data);
	  for (int i = 0; i < outliers.numInstances(); i++)
	  {
		  pred = (int)evaluateModelOnce(newClassifier, outliers.instance(i));
		  System.out.print(outliers.instance(i).value(0) + "\t" + data.classAttribute().value((int)outliers.instance(i).classValue()) + "\t" + data.classAttribute().value(pred) + "\tlast" + "\n");
	  }
      m_NumFolds = numFolds;
   }
   
   public void toSummaryStringCrossValidateModelOutlierOverSample(Classifier classifier, Instances data, 
         int numFolds, Random random) 
   throws Exception 
   {
   	data = new Instances(data);
   	Instances[] classes = new Instances[data.numClasses()];
   	/*for (int i = 0; i < classes.length; i++)
   	{
   		classes[i] = new Instances(data,0);
   	}*/
   	//Instances data2 = new Instances(data);
   	Instances outliers = new Instances(data, 0);
   	for (int i = 0; i < data.numInstances(); i++)
   	{
   		//System.out.println(data.instance(i).value(1) == 0);
   		if (data.instance(i).value(1)==0)
   		{
   			System.out.println(data.instance(i).value(0));
   			outliers.add(data.instance(i));
      		data.delete(i);
      		i--;
   		}
   		/*else
   		{
   			//System.out.println(classes[0].numInstances());
   			classes[(int)data.instance(i).value(data.instance(i).classIndex())].add(data.instance(i));
   		}*/
   	}
   	
   	/*int max = 0;
   	int maxClass = 0;
   	for (int i = 0; i < classes.length; i++)
   	{
   		if (classes[i].numInstances() > max)
   		{
   			max = classes[i].numInstances();
   			maxClass = i;
   		}
   	}
   	//System.out.println(max + " " + maxClass);
   	
   	for (int i = 0; i < classes.length; i++)
   	{
   		int j = 0;
   		while (classes[i].numInstances() < max && classes[i].numInstances() > 0)
   		{
   			classes[i].add(classes[i].instance(j % classes[i].numInstances()));
   			j++;
   			data.add(classes[i].instance(j % classes[i].numInstances()));
   		}
   	}*/
   	//System.out.println(data.numInstances());
   	//if (true)
   	//return;
   	
   	
   	if (data.numInstances() < numFolds)
   	{
   		numFolds=data.numInstances();
   	}
	  // Make a copy of the data we can reorder
   	//System.out.println(data.numAttributes());
	  
	  data.randomize(random);
	  //data2.randomize(random);
	  int pred;
	  if (data.classAttribute().isNominal()) 
	  {
         data.stratify(numFolds);
	  }

	  /*if (data2.classAttribute().isNominal()) 
	  {
         data2.stratify(numFolds);
	  }*/
	  
	  System.out.print("Inst#\tActual\tPredicted\tfold#\n");
	  // Do the folds
	  for (int i = 0; i < numFolds; i++) 
	  {
        Instances train = data.trainCV(numFolds, i, random);
        for (int j = 0; j < classes.length; j++)
     	{
     		classes[j] = new Instances(data,0);
     	}
        for (int j = 0; j < train.numInstances(); j++)
        {
      	  classes[(int)train.instance(j).value(train.classIndex())].add(train.instance(j));
        }
        
        int max = 0;
     	int maxClass = 0;
     	for (int j = 0; j < classes.length; j++)
     	{
     		if (classes[j].numInstances() > max)
     		{
     			max = classes[j].numInstances();
     			maxClass = j;
     		}
     	}
     	System.out.println(max + " " + maxClass);
     	//if (true)
//     		return;
     	
     	for (int k = 0; k < classes.length; k++)
     	{
     		int j = 0;
     		while (classes[k].numInstances() < max && classes[k].numInstances() > 0)
     		{
     			classes[k].add(classes[k].instance(j % classes[k].numInstances()));
     			j++;
     			train.add(classes[k].instance(j % classes[k].numInstances()));
     		}
     	}
        
	     setPriors(train);
/*	     for (int j = 0; j < train.numInstances(); j++)
	     {
	   	  System.out.println(train.instance(j).value(0));
	     }*/
	     Classifier copiedClassifier = Classifier.makeCopy(classifier);
	     copiedClassifier.buildClassifier(train);
	     Instances test = data.testCV(numFolds, i);
        //System.out.println(test.instance(0).toString(0));
	     //evaluateModel(copiedClassifier, test);
	     for (int j = 0; j < test.numInstances(); j++)
	     {
	   	  //write to file here
	   	  pred = (int)evaluateModelOnce(copiedClassifier, test.instance(j));
	   	  System.out.print(test.instance(j).value(0) + "\t" + data.classAttribute().value((int)test.instance(j).classValue()) + "\t" + data.classAttribute().value(pred) + "\t" + i + "\n");
	   	  /*if (pred != test.instance(j).classValue())
	   	  {
	   		  pred = (int)test.instance(j).value(0);
	   		  if (!wrongPredictions.containsKey(pred))
	   		  {
	   			  wrongPredictions.put(pred, 1);
	   		  }
	   		  else
	   		  {
	   			  wrongPredictions.put(pred, wrongPredictions.get(pred) + 1);
	   		  }
	   		  
	   	  }*/
	   	  
	     }
	     
	  }
	  Classifier newClassifier = Classifier.makeCopy(classifier);
	  newClassifier.buildClassifier(data);
	  for (int i = 0; i < outliers.numInstances(); i++)
	  {
		  pred = (int)evaluateModelOnce(newClassifier, outliers.instance(i));
		  System.out.print(outliers.instance(i).value(0) + "\t" + data.classAttribute().value((int)outliers.instance(i).classValue()) + "\t" + data.classAttribute().value(pred) + "\tlast" + "\n");
	  }
      m_NumFolds = numFolds;
   }
   
   /**
    * Overwrites the croosValidateModel function from the base class. This is 
    * done to have more control over the evaluation.
    */
   public void toSummaryStringCrossValidateModel(Classifier classifier, Instances data, 
		                          int numFolds, Random random) 
   throws Exception 
   {
      	
	  // Make a copy of the data we can reorder
     //System.out.println(data.numAttributes());
     data = new Instances(data);
	  data.randomize(random);
	   
	  double pred;
	  if (data.classAttribute().isNominal()) 
	  {
         data.stratify(numFolds);
	  }
	  
	  System.out.print("Inst#\tActual\tPredicted\tfold#\n");
	  // Do the folds
	  for (int i = 0; i < numFolds; i++) 
	  {
        Instances train = data.trainCV(numFolds, i, random);
	     setPriors(train);
	     Classifier copiedClassifier = Classifier.makeCopy(classifier);
	     copiedClassifier.buildClassifier(train);
	     Instances test = data.testCV(numFolds, i);
        //System.out.println(test.instance(0).toString(0));
	     //evaluateModel(copiedClassifier, test);
	     for (int j = 0; j < test.numInstances(); j++)
	     {
	   	  //write to file here
	   	  //System.out.println(test.instance(j).classValue());
	   	  if (data.classAttribute().isNumeric())
	   	  {
		   	  pred = evaluateModelOnce(copiedClassifier, test.instance(j));
	   		  System.out.print(test.instance(j).value(0) + "\t" + test.instance(j).classValue() + "\t" + pred + "\t" + i + "\n");
	   	  }
	   	  else
	   	  {
		   	  pred = evaluateModelOnce(copiedClassifier, test.instance(j));
	   		  System.out.print(test.instance(j).value(0) + "\t" + data.classAttribute().value((int)test.instance(j).classValue()) + "\t" + data.classAttribute().value((int)pred) + "\t" + i + "\n");
	   	  }
	   	  /*if (pred != test.instance(j).classValue())
	   	  {
	   		  pred = (int)test.instance(j).value(0);
	   		  if (!wrongPredictions.containsKey(pred))
	   		  {
	   			  wrongPredictions.put(pred, 1);
	   		  }
	   		  else
	   		  {
	   			  wrongPredictions.put(pred, wrongPredictions.get(pred) + 1);
	   		  }
	   		  
	   	  }*/
	   	  
	     }
	     
	  }
      m_NumFolds = numFolds;
   }
   
   public void toSummaryStringCrossValidateModelNormalizedScores(Classifier classifier, Instances data, 
         int numFolds, Random random) 
   throws Exception 
   {

   	// Make a copy of the data we can reorder
   	//System.out.println(data.numAttributes());
   	data = new Instances(data);
   	data.randomize(random);

   	double[] probs;
   	if (data.classAttribute().isNominal()) 
   	{
   		data.stratify(numFolds);
   	}

   	System.out.print("Inst#\tNormScore#\n");
   	// Do the folds
   	for (int i = 0; i < numFolds; i++) 
   	{
   		Instances train = data.trainCV(numFolds, i, random);
   		setPriors(train);
   		Classifier copiedClassifier = Classifier.makeCopy(classifier);
   		copiedClassifier.buildClassifier(train);
   		Instances test = data.testCV(numFolds, i);

   		for (int j = 0; j < test.numInstances(); j++)
   		{
   			probs = copiedClassifier.distributionForInstance(test.instance(j));
   			NumberFormat format = new DecimalFormat("#.##########");
   			String f = format.format(1-probs[(int)test.instance(j).classValue()]);
   			System.out.print(test.instance(j).value(0) + "\t" + f + "\n");
   		}
   	}
   	m_NumFolds = numFolds;
   }
   
   /**
    * Overwrites the croosValidateModel function from the base class. This is 
    * done to have more control over the evaluation.
    */
   public void crossValidateModel(Classifier classifier, Instances data, 
		                          int numFolds, Random random, String dir, String fileName) 
   throws Exception 
   {
   	//String homeDir = "/home/msmith/Desktop/Research/results/";
     if (!(new File(dir).exists()))
     {
   	  new File(dir).mkdir();
     }
     FileWriter file = new FileWriter(dir + fileName);
     BufferedWriter out = new BufferedWriter(file);
     boolean output = false;
     
	  // Make a copy of the data we can reorder
	  data = new Instances(data);
	  data.randomize(random);
	  ArffSaver saver = new ArffSaver();
	  int pred;
	  if (data.classAttribute().isNominal()) 
	  {
         data.stratify(numFolds);
	  }
	  
	  out.write("Inst#\tActual\tPredicted\tfold#\n");
	  // Do the folds
	  for (int i = 0; i < numFolds; i++) 
	  {
        Instances train = data.trainCV(numFolds, i, random);
	     setPriors(train);
	     Classifier copiedClassifier = Classifier.makeCopy(classifier);
	     copiedClassifier.buildClassifier(train);
	     Instances test = data.testCV(numFolds, i);
        //System.out.println(test.instance(0).toString(0));
	     //evaluateModel(copiedClassifier, test);
	     for (int j = 0; j < test.numInstances(); j++)
	     {
	   	  //write to file here
	   	  pred = (int)evaluateModelOnce(copiedClassifier, test.instance(j));
	   	  out.write(test.instance(j).value(0) + "\t" + data.classAttribute().value((int)test.instance(j).classValue()) + "\t" + data.classAttribute().value(pred) + "\t" + i + "\n");
	   	  if (pred != test.instance(j).classValue())
	   	  {
	   		  pred = (int)test.instance(j).value(0);
	   		  if (!wrongPredictions.containsKey(pred))
	   		  {
	   			  wrongPredictions.put(pred, 1);
	   		  }
	   		  else
	   		  {
	   			  wrongPredictions.put(pred, wrongPredictions.get(pred) + 1);
	   		  }
	   		  
	   	  }
	   	  
	     }
	     if (output)
	     {
	     saver.setInstances(test);
        saver.setFile(new File("/tmp/temp.arff"));
        saver.writeBatch();

	/*     System.out.println(printClassifications(copiedClassifier, train, 
	   		  new DataSource("/tmp/temp.arff"), data.numAttributes(), 
	   		  new Range("1"), true));
	   		  */
	     }
	  }
	  out.close();
      m_NumFolds = numFolds;
   }

   /**
    *  This only works with a Multiclass classifier
    * @param classifier
    * @param data
    * @param numFolds
    * @param random
    * @throws Exception
    */
	public void toSummaryStringCrossValidateModelLogReg(FilteredMultiClassClassifier classifier,
			Instances data, int numFolds, Random random) throws Exception  
   {

   	// Make a copy of the data we can reorder
   	//System.out.println(data.numAttributes());
   	data = new Instances(data);
   	data.randomize(random);

   	double[][] probs;
   	if (data.classAttribute().isNominal()) 
   	{
   		data.stratify(numFolds);
   	}

   	System.out.print("Inst#\tNormScore#\n");
   	// Do the folds
   	for (int i = 0; i < numFolds; i++) 
   	{
   		Instances train = data.trainCV(numFolds, i, random);
   		setPriors(train);
   		FilteredMultiClassClassifier copiedClassifier = (FilteredMultiClassClassifier) Classifier.makeCopy(classifier);
   		copiedClassifier.buildClassifier(train);
   		Instances test = data.testCV(numFolds, i);

   		//get Binary Scores
   		probs = copiedClassifier.calibratedDistributionForTestInstances(test);
   		for (int j = 0; j < test.numInstances(); j++)
   		{
   			//binProbs[j] = copiedClassifier.binaryDistributionForInstance(test.instance(j));
   			//System.out.print(test.instance(j).value(0) + "\t" + probs[j] + "\n");
   			//for (int k = 0; k < 3; k++)
   				//System.out.print(test.instance(j).value(0) + "\t" + probs[k][j] + "\n");
   			NumberFormat format = new DecimalFormat("#.##########");
   			String f = "DUDE";
   			if (test.classAttribute().numValues() == 2)
   			{
   				if (test.instance(j).classValue() == 0.0)
   					f = format.format(1-probs[0][j]);
   				else
   					f = format.format(probs[0][j]);
   			}
   			else
   				f = format.format(1-probs[(int)test.instance(j).classValue()][j]);
   			System.out.print(test.instance(j).value(0) + "\t" + f + " " + test.instance(j).classValue() + "\n");
   		}
   	}
   	m_NumFolds = numFolds;
		
	}

}
