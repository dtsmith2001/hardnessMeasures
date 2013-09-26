package library;

//import java.io.BufferedWriter;
//import java.io.FileWriter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
import java.util.Vector;
//import java.util.Vector;

//import weka.classifiers.lazy.IBk;
import weka.core.FastVector;
import weka.core.Instance;
//import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class MikeIBk extends weka.classifiers.lazy.IBk 
{
	/** for serialization. */
	static final long serialVersionUID = -3080186098777067172L;
	
	protected Instances border;
	protected Instances notBorder;
	protected Instances noise;
	protected Instances notNoise;
	protected Vector<Integer> noiseInstances;
	protected Vector<Integer> borderInstances;
	//protected String borderFileName;
	//protected String noiseFileName;
	
	public MikeIBk()
	{
		super();
		//fileName = "";
		//System.out.println("testing This");
	}
	
	public double getNumNeighbors(Instance instance) throws Exception
	{
		int count = 0;
		double classType = instance.classValue();
		double percent = 0.0;
                double overallPercent = 0.0;
                int num = 0;

		if(m_Train.numInstances() > 0)
		{
			Instances neighbors = m_NNSearch.kNearestNeighbours(instance, m_kNN + 1 );

                for (int k = 1; k < neighbors.numInstances(); k+=2)
                {
                count = 0;
                percent = 0.0;
   		for (int j = 1; j < neighbors.numInstances() && j < k+1; j++) //neighbors.numInstances(); j++)
   		{
   			if (classType == neighbors.instance(j).classValue())
   			{
   				count++;
   			}
   		}
//                int val = (k > neighbors.numInstances()) ? neighbors.numInstances() - 1 : k;
   		percent = count * 1.0 / k; //(neighbors.numInstances()-1);
//System.out.print(percent + " ");
                overallPercent += percent;
                num++;
                }
		}
		return overallPercent / num;
	}
	
	public void findNoiseBorder(Instances instances) throws Exception
	{
		//Get the attributes
		FastVector attr = new FastVector();
   	for (int i = 0; i < instances.numAttributes(); i++)
   	{
   		attr.addElement(instances.attribute(i));
   		//System.out.println(instances.attribute(i).name());
   	}
   	
   	//Create instances vectors
		borderInstances = new Vector<Integer>();
   	noiseInstances = new Vector<Integer>();
   	border = new Instances("Border", attr, 0);
   	noise = new Instances("Border", attr, 0);
   	notBorder = new Instances("Border", attr, 0);
   	notNoise = new Instances("Border", attr, 0);
   	
   	//Build the classifier to find outliers
   	//Filter the data set for the ID attribute
   	String[] options = new String[2];
   	options[0] = "-R";                                    // "range"
   	options[1] = "1";                                     // first attribute
   	Remove remove = new Remove();                         // new instance of filter
   	remove.setOptions(options);                           // set options
   	remove.setInputFormat(instances);                          // inform filter about dataset AFTER setting options
   	Instances newData = Filter.useFilter(instances, remove);   // apply filter
   	newData.setClassIndex(newData.numAttributes() - 1);
   	
   	buildClassifier(newData);
   	
   	HashMap<Double, Integer> classes = new HashMap<Double, Integer>();
		int temp = 0;

   	for (int i = 0; i < instances.numInstances(); i++)
   	{
   		classes.clear();
   		m_NNSearch.setInstances(m_Train);
   		Instances neighbours = m_NNSearch.kNearestNeighbours(newData.instance(i), m_kNN + 1 );
   		   		
   		for (int j = 1; j < neighbours.numInstances(); j++)
   		{
   			if (!(classes.containsKey(neighbours.instance(j).classValue())))
   			{
   				temp = 1;
   			}
   			else
   			{
   				temp = classes.get(neighbours.instance(j).classValue()) + 1;
   			}
   			classes.put(neighbours.instance(j).classValue(), temp);
   		}
   		
   		//System.out.println((i+1) + " " + temp + " " + classes + " " + neighbours.numInstances());
   		
   		Set<Double> keys = classes.keySet();
   		
   		if ((!keys.contains(instances.instance(i).classValue())) || classes.get(instances.instance(i).classValue()) < m_kNN/2.0)
   		{
   			//System.out.println((i+1) + " " + classes.get(instances.instance(i).classValue()) + " " + temp + " " + m_kNN/2.0);
   			noise.add(instances.instance(i));
   			noiseInstances.add((int)instances.instance(i).value(0));
   		}
   		else
   		{
   			notNoise.add(instances.instance(i));
   		}
   	}
   	
   	//retrain on the notNoisy instances
   	remove = new Remove();                         // new instance of filter
   	remove.setOptions(options);                           // set options
   	remove.setInputFormat(notNoise);                          // inform filter about dataset AFTER setting options
   	newData = Filter.useFilter(notNoise, remove);   // apply filter
   	newData.setClassIndex(newData.numAttributes() - 1);
   	
   	buildClassifier(newData);
   	
   	for (int i = 0; i < notNoise.numInstances(); i++)
   	{
   		classes.clear();
   		m_NNSearch.setInstances(m_Train);
   		Instances neighbours = m_NNSearch.kNearestNeighbours(newData.instance(i), m_kNN + 1);
   		   		
   		for (int j = 1; j < neighbours.numInstances(); j++)
   		{
   			//System.out.println(neighbours.instance(j).classValue());
   			if (!(classes.containsKey(neighbours.instance(j).classValue())))
   			{
   				temp = 1;
   			}
   			else
   			{
   				temp = classes.get(neighbours.instance(j).classValue()) + 1;
   			}
   			classes.put(neighbours.instance(j).classValue(), temp);
   		}
   		
   		Set<Double> keys = classes.keySet();
   		Iterator<Double> iter = keys.iterator();
   		double min = 99;
   		
   		while (iter.hasNext())
   		{
   			double value = classes.get(iter.next());
   			if (value < min)
   			{
   				min = value;
   			}
   		}

   		if (min < m_kNN && min > 0)
   		{
   			//System.out.println(i + " " + borderInstances.size());
   			//out.write(i + "\n");
   			//System.out.println(notNoise.instance(i).value(0) + " " + notNoise.instance(i));
   			border.add(notNoise.instance(i));
   			borderInstances.add((int)notNoise.instance(i).value(0));
   			//System.out.println((int)instances.instance(i).value(0));
   		}
   		else
   		{
   			notBorder.add(notNoise.instance(i));
   		}
   		
   	}
	}
	
   /*public void findNoise(Instances instances) throws Exception
   {
   	
   	//Remove remove = new Remove();
   	//remove
   	
   	

   	
   	HashMap<Double, Integer> classes = new HashMap<Double, Integer>();
		int temp;
		//int max = 2;

		//FileWriter file = new FileWriter(fileName);
	   //BufferedWriter out = new BufferedWriter(file);
   	for (int i = 0; i < instances.numInstances(); i++)
   	{
   		//max = 99;
   		//double classValue = instances.instance(i).classValue();
   		classes.clear();
   		m_NNSearch.setInstances(m_Train);
   		Instances neighbours = m_NNSearch.kNearestNeighbours(newData.instance(i), m_kNN + 1);
   		
   		//if (i ==106 || i == 70)
   		//System.out.println((i+1) + " " + newData.instance(i));
   		
   		for (int j = 1; j < neighbours.numInstances(); j++)
   		{
   			
   			if (classValue != neighbours.instance(j).classValue())
   			{
   				max++;
   			}
   			
   		   //if (i == 106 || i == 70)
   			//System.out.println("DUDE " + (i+1) + " " + neighbours.instance(j).toString() + " " + neighbours.instance(j).classValue());
   			if (!(classes.containsKey(neighbours.instance(j).classValue())))
   			{
   				//classes.put(neighbours.instance(j).classValue(), 1);
   				temp = 1;
   			}
   			else
   			{
   				temp = classes.get(neighbours.instance(j).classValue()) + 1;
   			}
   			
   			if (temp < max)
   			{
   				max = temp;
   			}
   			
   			classes.put(neighbours.instance(j).classValue(), temp);
   			
   			
   			if (max > m_kNN / 2)
   			{
   				border.add(instances.instance(i));
   				j = neighbours.numInstances();
   			}
   			
   		}
   		
   		Set<Double> keys = classes.keySet();
   		Iterator<Double> iter = keys.iterator();
   		double min = 99;
   		//if (keys.contains(instances.instance(i).classValue()))
   		if ((!keys.contains(instances.instance(i).classValue())) || classes.get(instances.instance(i).classValue()) < m_kNN/2.0)
   		{
   			System.out.println((i+1) + " " + classes.get(instances.instance(i).classValue()));
   			//instances.delete(index);
   			noise.add(instances.instance(i));
   			noiseInstances.add((int)instances.instance(i).value(0));
   		}
   		else
   		{
   			notNoise.add(instances.instance(i));
   		}
   		
   		while (iter.hasNext())
   		{
   			double value = classes.get(iter.next());
   			if (value < min)
   			{
   				min = value;
   			}
   		}

   		if (min < m_kNN && min > 0)
   		{
   			//System.out.println(i + " " + borderInstances.size());
   			//out.write(i + "\n");
   			border.add(instances.instance(i));
   			borderInstances.add((int)instances.instance(i).value(0));
   			//System.out.println((int)instances.instance(i).value(0));
   		}
   		else
   		{
   			notBorder.add(instances.instance(i));
   		}
   		
   		
   		if (classes.size() > 1)
   		{
   			border.add(instances.instance(i));
   		}
   		
   	}
   	//out.close();
   	//System.out.println("Noise: " + noise.numInstances());
   	for (int k = 0; k < noise.numInstances(); k++)
   	{
   		System.out.println(noise.instance(k));
   	}
   }
   
   public void findBorder(Instances instances) throws Exception
   {
   	//Filter the data set for the ID attribute
   	String[] options = new String[2];
   	options[0] = "-R";                                    // "range"
   	options[1] = "1";                                     // first attribute
   	Remove remove = new Remove();                         // new instance of filter
   	remove.setOptions(options);                           // set options
   	remove.setInputFormat(instances);                          // inform filter about dataset AFTER setting options
   	Instances newData = Filter.useFilter(instances, remove);   // apply filter
   	newData.setClassIndex(newData.numAttributes() - 1);
   	//Remove remove = new Remove();
   	//remove
   	
   	FastVector attr = new FastVector();
   	for (int i = 0; i < instances.numAttributes(); i++)
   	{
   		attr.addElement(instances.attribute(i));
   		//System.out.println(instances.attribute(i).name());
   	}

   	borderInstances = new Vector<Integer>();
   	noiseInstances = new Vector<Integer>();
   	border = new Instances("Border", attr, 0);
   	noise = new Instances("Border", attr, 0);
   	notBorder = new Instances("Border", attr, 0);
   	notNoise = new Instances("Border", attr, 0);
   	HashMap<Double, Integer> classes = new HashMap<Double, Integer>();
		int temp;
		//int max = 2;

		//FileWriter file = new FileWriter(fileName);
	   //BufferedWriter out = new BufferedWriter(file);
   	for (int i = 0; i < instances.numInstances(); i++)
   	{
   		//max = 99;
   		//double classValue = instances.instance(i).classValue();
   		classes.clear();
   		m_NNSearch.setInstances(m_Train);
   		Instances neighbours = m_NNSearch.kNearestNeighbours(newData.instance(i), m_kNN + 1);
   		
   		//if (i ==106 || i == 70)
   		//System.out.println((i+1) + " " + newData.instance(i));
   		
   		for (int j = 1; j < neighbours.numInstances(); j++)
   		{
   			
   			if (classValue != neighbours.instance(j).classValue())
   			{
   				max++;
   			}
   			
   		   //if (i == 106 || i == 70)
   			//System.out.println("DUDE " + (i+1) + " " + neighbours.instance(j).toString() + " " + neighbours.instance(j).classValue());
   			if (!(classes.containsKey(neighbours.instance(j).classValue())))
   			{
   				//classes.put(neighbours.instance(j).classValue(), 1);
   				temp = 1;
   			}
   			else
   			{
   				temp = classes.get(neighbours.instance(j).classValue()) + 1;
   			}
   			
   			if (temp < max)
   			{
   				max = temp;
   			}
   			
   			classes.put(neighbours.instance(j).classValue(), temp);
   			
   			
   			if (max > m_kNN / 2)
   			{
   				border.add(instances.instance(i));
   				j = neighbours.numInstances();
   			}
   			
   		}
   		
   		Set<Double> keys = classes.keySet();
   		Iterator<Double> iter = keys.iterator();
   		double min = 99;
   		//if (keys.contains(instances.instance(i).classValue()))
   		if ((!keys.contains(instances.instance(i).classValue())) || classes.get(instances.instance(i).classValue()) < m_kNN/2.0)
   		{
   			System.out.println((i+1) + " " + classes.get(instances.instance(i).classValue()));
   			//instances.delete(index);
   			noise.add(instances.instance(i));
   			noiseInstances.add((int)instances.instance(i).value(0));
   		}
   		else
   		{
   			notNoise.add(instances.instance(i));
   		}
   		
   		while (iter.hasNext())
   		{
   			double value = classes.get(iter.next());
   			if (value < min)
   			{
   				min = value;
   			}
   		}

   		if (min < m_kNN && min > 0)
   		{
   			//System.out.println(i + " " + borderInstances.size());
   			//out.write(i + "\n");
   			border.add(instances.instance(i));
   			borderInstances.add((int)instances.instance(i).value(0));
   			//System.out.println((int)instances.instance(i).value(0));
   		}
   		else
   		{
   			notBorder.add(instances.instance(i));
   		}
   		
   		
   		if (classes.size() > 1)
   		{
   			border.add(instances.instance(i));
   		}
   		
   	}
   	//out.close();
   	//System.out.println("Noise: " + noise.numInstances());
   	for (int k = 0; k < noise.numInstances(); k++)
   	{
   		System.out.println(noise.instance(k));
   	}
   }*/
   
   public Instances getNoise() throws Exception
   {
   	if (noise == null)
   	{
   		throw new Exception("Noise is null");
   	}
   	return noise;
   }

   public Instances getNotNoise() throws Exception
   {
   	if (notNoise == null)
   	{
   		throw new Exception("notNoise is null");
   	}
   	return notNoise;
   }   
   
   public Instances getNotBorder() throws Exception
   {
   	if (notBorder == null)
   	{
   		throw new Exception("notBorder is null");
   	}
   	return notBorder;
   }
   
   public Instances getBorder() throws Exception
   {
   	if (border == null)
   	{
   		throw new Exception("Border is null");
   	}
   	return border;
   }
   
   public Vector<Integer> getBorderInstances()
   {
   	return borderInstances;
   }
   
   public Vector<Integer> getNoiseInstances()
   {
   	return noiseInstances;
   }
}
