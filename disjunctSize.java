import library.MikeJ48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;



public class disjunctSize// extends J48 
{
	public double getBagDist (int index)
	{
		//super.m_root
		return 0.0;
	}
	public static void main(String[] args) throws Exception
   {
      String dataset; 		//The name of the data set to use.  It is passed in.
      DataSource source;	//Source of the dataset
      Instances data=null;		//Original data
      int classIndex;		//Index of the class attribute
      boolean f = Utils.getFlag("f", args);
      int [] id;
      Remove r;
      MikeJ48 classify = null;
		
      if (args.length < 1)
      {
         throw new Exception("Usage: disjunctSize {Dataset} [-p] [-c classIndex]\n");
      }
		
      dataset = args[0];
		
      //Initialize the dataset
      source = new DataSource(dataset);
      data = source.getDataSet();
      id = new int[data.numInstances()];
      if (f)
      {
         for (int i = 0; i < data.numInstances(); i++)
         {
            id[i]=(int)data.instance(i).value(0);
         }

         r = new Remove();
         String[] opts=new String[2];
         opts[0] = "-R";
         opts[1] = "first";
         r.setInputFormat(data);
         r.setOptions(opts);
         data=Filter.useFilter(data, r);
      }
      else
         for (int i = 0; i < data.numInstances(); i++)
            id[i] = i;
		
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
		
      int pruneIdx = Utils.getOptionPos('p', args);
      String[] options;
      
      if (pruneIdx > -1)
      {
      	options = new String[4];
      	options[0] = "-C";
      	options[1] = "0.25";
      	options[2] = "-M";
      	options[3] = "2";
        classify = new MikeJ48();
      }
      else
      {
      	options = new String[3];
      	options[0] = "-U";
      	options[1] = "-M";
      	options[2] = "1";
        classify = new MikeJ48();
      }
      //Create the Classifier
      classify.setOptions(options);
      
      classify.buildClassifier(data); 
      
      for (int inst = 0; inst < data.numInstances(); inst++)
      {
      	 System.out.print(id[inst] + " " + (int)data.instance(inst).classValue());
         double [] disjunct = classify.disjunctSize(data.instance(inst));
         for (int i=0; i < disjunct.length; i++)
         {
      	   System.out.print(" " + disjunct[i]);
         }
         System.out.println();
      }

      String filteredInstances = Utils.getOption("test", args);
      if (filteredInstances.length() != 0)
      {
          DataSource newSource = new DataSource(filteredInstances);
          Instances newData = newSource.getDataSet();
          if (f)
          {
             id = new int[newData.numInstances()];
             for (int i = 0; i < newData.numInstances(); i++)
             {
                id[i]=(int)newData.instance(i).value(0);
             }

             r = new Remove();
             String[] opts=new String[2];
             opts[0] = "-R";
             opts[1] = "first";
             r.setInputFormat(newData);
             r.setOptions(opts);
             newData=Filter.useFilter(newData, r);
             newData.setClassIndex(data.numAttributes() - 1);
 
             for (int inst = 0; inst<newData.numInstances(); inst++)
             {
                System.out.print(id[inst] + " " + (int)newData.instance(inst).classValue());
                double [] disjunct = classify.disjunctSize(newData.instance(inst));

                for (int i=0; i < disjunct.length; i++)
                {
                   System.out.print(" " + disjunct[i]);
                }
                System.out.println();
             }
          }
          else
             for (int inst = 0; inst<newData.numInstances(); inst++)
             {
                 System.out.print(id[inst] + " " + (int)newData.instance(inst).classValue());
                double [] disjunct = classify.disjunctSize(newData.instance(inst));
                for (int i=0; i < disjunct.length; i++)
                {
                   System.out.print(" " + disjunct[i]);
                }
                System.out.println();
             }

       }


      System.out.println(classify.toSummaryString());
   }
}
