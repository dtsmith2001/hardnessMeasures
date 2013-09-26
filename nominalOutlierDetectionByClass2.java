import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddID;

/**
 *  Find the outliers using the interquartile range filter provided by Weka
 * @author msmith
 *
 */
public class nominalOutlierDetectionByClass2 {
	public static void main(String[] args) throws Exception
	{
		String dataset;		//Name of the dataset to use
      DataSource source;	//Source of the dataset
      Instances data;		//Original data
      int classIndex;		//Index of the class attribute
      int[] id;
      boolean f = Utils.getFlag("f", args);

		//For testing purposes
		//System.out.println("Testing Here Dude" + args.length);
		
		//Get the learning algorithm and dataset name 
      //Make sure there are at least two arguments
   	if (args.length < 1)
   	{
   		throw new Exception("Usage: nominalOutlierDetectioByClass {Dataset} [-c classIndex]\n");
   	}
		dataset = args[0];
		
		//Initialize the dataset
		source = new DataSource(dataset);
		data = source.getDataSet();
                id = new int[data.numInstances()];
		
		//Add an ID attribute to keep track of the instances
      if (f)
      {
         for (int i = 0; i < data.numInstances(); i++)
            id[i] = (int)data.instance(i).value(0);
      }
      else if (data.attribute(1).name().toLowerCase() != "id") // || data.attribute(1).isNumeric())
      {
         for(int i = 0; i < data.numInstances(); i++)
            id[i]=i;
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
      ignore = "1," + ignore;
      
		//Create the Classifier
      Classifier classify = new FilteredClassifier();
      
      String[] classifier = new String[6];
      classifier[0] = "-F";
	  	classifier[1] = "weka.filters.unsupervised.attribute.Remove -R 1";
	  	classifier[2] = "-W";
	  	//classifier[3] = "NBOutliers";
   	classifier[3] = "weka.classifiers.bayes.NaiveBayes";
   	classifier[4] = "--";
   	//classifier[5] = "-K";
   	classifier[5] = "-D";
   	classify.setOptions(classifier);
   	
   	//classify.buildClassifier(data);
   	
   	double max;
   	double next;
   	double maxClass;
   	double[] ave = new double[data.numClasses()];
   	int[] total = new int[data.numClasses()];
//   	for (int i = 0; i < data.numClasses(); i++)
//   	{
//   		ave[i] = 0;
//   		total[i]=0;
//   	}
   	Instances newData;
   	//dataset = dataset.substring(dataset.lastIndexOf("/") + 1, dataset.lastIndexOf(".arff")); 
   	for (int i = 0; i < data.numInstances(); i++)
   	{
   		//System.out.println(data.instance(i).toString());
         newData = new Instances(data);
      	newData.delete(i);
      	//System.out.println(data.numInstances() + " " + newData.numInstances());
      	classify.buildClassifier(newData);
   		max = 0;
   		next = 0;
   		maxClass = 0;
   		System.out.print(id[i]);
   		double [] temp = classify.distributionForInstance(data.instance(i));
   		//for (int j = 0; j < temp.length - 1; j++)
   		for (int j = 0; j < temp.length; j++)
   		{
   			//ave[j]+=temp[j];
   			if (temp[j] > max)
   			{
   				next = max;
   				max = temp[j];
   				maxClass = j;
   			}
   			else if (temp[j] > next)
   			{
   				next = temp[j];
   			}
   			System.out.print(" " + temp[j]);
   		}
   		//System.out.println();
   		double classVal = data.instance(i).classValue();
   		System.out.println(" " + classVal + " " + maxClass + " " + max + " " + (max - next));
   		
   		ave[(int)(classVal)]+=temp[(int) classVal];
   		total[(int)classVal]+=1;
   		//System.out.println(classify.toString());
   	}

        String filteredInstances = Utils.getOption("test", args);
        if (filteredInstances.length() != 0)
        {
           DataSource newSource = new DataSource(filteredInstances);
           newData = newSource.getDataSet();
           newData.setClassIndex(newData.numAttributes() - 1);
           id = new int[newData.numInstances()];
           if (f)
              for (int i = 0; i < newData.numInstances(); i++)
                 id[i]=(int)newData.instance(i).value(0);
           else
              for (int i = 0; i < newData.numInstances(); i++)
                 id[i]=i;
           
           classify.buildClassifier(data);
           for (int i = 0; i < newData.numInstances(); i++)
           {
              max = 0;
              next = 0;
              maxClass = 0;
              System.out.print(id[i]);
              double [] temp = classify.distributionForInstance(newData.instance(i));
              
              for (int j = 0; j < temp.length; j++)
              {
                 if (temp[j] > max)
                 {
                    next = max;
                    max = temp[j];
                    maxClass = j;
                 }
                 else if (temp[j] > next)
                 {
                    next = temp[j];
                 }
                 System.out.print(" " + temp[j]);
              }
             
              double classVal = newData.instance(i).classValue();
              System.out.println(" " + classVal + " " + maxClass + " " + max + " " + (max - next));
        }

        }
   	for (int i = 0; i < ave.length; i++)
   	System.out.println("Average " + i + " " + ave[i] / total[i]);
   }
}
