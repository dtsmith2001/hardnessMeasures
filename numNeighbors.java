/**
 * Michael R. Smith 2010
 */


import library.MikeIBk;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

class numNeighbors
{
   public static void main(String[] args)
   {
   	//Initialize the data set
      DataSource source;
      Instances data;

      //Get the data set name
      String dataset = args[0];
      String numNeighbors = args[1];
      Remove r = null;
      int[] id = null;
      boolean f = false;
      
      try
      {
         f = Utils.getFlag("f", args);
         source = new DataSource(dataset);
         data = source.getDataSet();
         data.setClassIndex(data.numAttributes() - 1);

         if (f)
         {
            id = new int[data.numInstances()];
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
   
         MikeIBk noise = new MikeIBk();
         String[] options = new String[6];
	      options[0] = "-K";
      	options[1] = numNeighbors;
      	options[2] = "-W";
      	options[3] = "0";
      	options[4] = "-A";
      	options[5] = "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"";
	
      	noise.setOptions(options);
      	noise.buildClassifier(data);
  
        if(f)    	
           for (int i = 0; i<data.numInstances(); i++)
           {
  	       		System.out.println(id[i] + " " + noise.getNumNeighbors(data.instance(i)));
      	   }
        else
      	   for (int i = 0; i<data.numInstances(); i++)
      	   {
  	    		System.out.println((i+1) + " " + noise.getNumNeighbors(data.instance(i)));
      	   }

        String filteredInstances = Utils.getOption("test", args);
        if (filteredInstances.length() != 0)
        {
            DataSource newSource = new DataSource(filteredInstances);
            Instances newData = newSource.getDataSet();
            newData.setClassIndex(data.numAttributes() - 1);
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

               for (int i = 0; i<newData.numInstances(); i++)
               {
                  data.add(newData.instance(i));
                  noise.buildClassifier(data);
                  System.out.println(id[i] + " " + noise.getNumNeighbors(data.lastInstance()));
                  data.delete(data.numInstances()-1);
               }
            }
            else
               for (int i = 0; i<newData.numInstances(); i++)
               {
                  data.add(newData.instance(i));
                  noise.buildClassifier(data);
                        System.out.println((i+1) + " " + noise.getNumNeighbors(newData.instance(i)));
                  data.delete(data.numInstances()-1);
               } 

         }
      }
      catch (Exception e)
      {
      	e.printStackTrace();
      	System.out.println("Something went wrong\n");
      }
      
   }
}
