/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    MultiClassClassifier.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 */

import weka.classifiers.Classifier;
import weka.classifiers.RandomizableSingleClassifierEnhancer;
import weka.classifiers.rules.ZeroR;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.MakeIndicator;
import weka.filters.unsupervised.instance.RemoveWithValues;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

/**
 <!-- globalinfo-start -->
 * A metaclassifier for handling multi-class datasets with 2-class classifiers. This classifier is also capable of applying error correcting output codes for increased accuracy.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -M &lt;num&gt;
 *  Sets the method to use. Valid values are 0 (1-against-all),
 *  1 (random codes), 2 (exhaustive code), and 3 (1-against-1). (default 0)
 * </pre>
 * 
 * <pre> -R &lt;num&gt;
 *  Sets the multiplier when using random codes. (default 2.0)</pre>
 * 
 * <pre> -P
 *  Use pairwise coupling (only has an effect for 1-against1)</pre>
 * 
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.functions.Logistic)</pre>
 * 
 * <pre> 
 * Options specific to classifier weka.classifiers.functions.Logistic:
 * </pre>
 * 
 * <pre> -D
 *  Turn on debugging output.</pre>
 * 
 * <pre> -R &lt;ridge&gt;
 *  Set the ridge in the log-likelihood.</pre>
 * 
 * <pre> -M &lt;number&gt;
 *  Set the maximum number of iterations (default -1, until convergence).</pre>
 * 
 <!-- options-end -->
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Len Trigg (len@reeltwo.com)
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 1.48 $
 */
public class MultiClassClassifier 
  extends RandomizableSingleClassifierEnhancer 
  implements OptionHandler {

  /** for serialization */
  static final long serialVersionUID = -3879602011542849141L;
  
  /** The classifiers. */
  private Classifier [] m_Classifiers;

  /** Use pairwise coupling with 1-vs-1 */
  private boolean m_pairwiseCoupling = false;

  /** Needed for pairwise coupling */
  private double [] m_SumOfWeights;

  /** The filters used to transform the class. */
  private Filter[] m_ClassFilters;

  /** ZeroR classifier for when all base classifier return zero probability. */
  private ZeroR m_ZeroR;

  /** Internal copy of the class attribute for output purposes */
  private Attribute m_ClassAttribute;
  
  /** A transformed dataset header used by the  1-against-1 method */
  private Instances m_TwoClassDataset;

  /** 
   * The multiplier when generating random codes. Will generate
   * numClasses * m_RandomWidthFactor codes
   */
  private double m_RandomWidthFactor = 2.0;

  /** The multiclass method to use */
  private int m_Method = METHOD_1_AGAINST_ALL;

  /** 1-against-all */
  public static final int METHOD_1_AGAINST_ALL    = 0;
  /** random correction code */
  public static final int METHOD_ERROR_RANDOM     = 1;
  /** exhaustive correction code */
  public static final int METHOD_ERROR_EXHAUSTIVE = 2;
  /** 1-against-1 */
  public static final int METHOD_1_AGAINST_1      = 3;
  /** The error correction modes */
  public static final Tag [] TAGS_METHOD = {
    new Tag(METHOD_1_AGAINST_ALL, "1-against-all"),
    new Tag(METHOD_ERROR_RANDOM, "Random correction code"),
    new Tag(METHOD_ERROR_EXHAUSTIVE, "Exhaustive correction code"),
    new Tag(METHOD_1_AGAINST_1, "1-against-1")
  };
    
  /**
   * Constructor.
   */
  public MultiClassClassifier() {
    
    m_Classifier = new weka.classifiers.functions.Logistic();
  }

  /**
   * String describing default classifier.
   * 
   * @return the default classifier classname
   */
  protected String defaultClassifierString() {
    
    return "weka.classifiers.functions.Logistic";
  }

  /** 
   * Interface for the code constructors 
   */
  private abstract class Code 
    implements Serializable, RevisionHandler {

    /** for serialization */
    static final long serialVersionUID = 418095077487120846L;
    
    /**
     * Subclasses must allocate and fill these. 
     * First dimension is number of codes.
     * Second dimension is number of classes.
     */
    protected boolean [][]m_Codebits;

    /** 
     * Returns the number of codes. 
     * @return the number of codes
     */
    public int size() {
      return m_Codebits.length;
    }

    /** 
     * Returns the indices of the values set to true for this code, 
     * using 1-based indexing (for input to Range).
     * 
     * @param which the index
     * @return the 1-based indices
     */
    public String getIndices(int which) {
      StringBuffer sb = new StringBuffer();
      for (int i = 0; i < m_Codebits[which].length; i++) {
        if (m_Codebits[which][i]) {
          if (sb.length() != 0) {
            sb.append(',');
          }
          sb.append(i + 1);
        }
      }
      return sb.toString();
    }

    /** 
     * Returns a human-readable representation of the codes. 
     * @return a string representation of the codes
     */
    public String toString() {
      StringBuffer sb = new StringBuffer();
      for(int i = 0; i < m_Codebits[0].length; i++) {
        for (int j = 0; j < m_Codebits.length; j++) {
          sb.append(m_Codebits[j][i] ? " 1" : " 0");
        }
        sb.append('\n');
      }
      return sb.toString();
    }
    
    /**
     * Returns the revision string.
     * 
     * @return		the revision
     */
    public String getRevision() {
      return RevisionUtils.extract("$Revision: 1.48 $");
    }
  }

  /** 
   * Constructs a code with no error correction 
   */
  private class StandardCode 
    extends Code {
    
    /** for serialization */
    static final long serialVersionUID = 3707829689461467358L;
    
    /**
     * constructor
     * 
     * @param numClasses the number of classes
     */
    public StandardCode(int numClasses) {
      m_Codebits = new boolean[numClasses][numClasses];
      for (int i = 0; i < numClasses; i++) {
        m_Codebits[i][i] = true;
      }
      //System.err.println("Code:\n" + this);
    }
    
    /**
     * Returns the revision string.
     * 
     * @return		the revision
     */
    public String getRevision() {
      return RevisionUtils.extract("$Revision: 1.48 $");
    }
  }

  /** 
   * Constructs a random code assignment 
   */
  private class RandomCode 
    extends Code {

    /** for serialization */
    static final long serialVersionUID = 4413410540703926563L;
    
    /** random number generator */
    Random r = null;
   
    /**
     * constructor
     * 
     * @param numClasses the number of classes
     * @param numCodes the number of codes
     * @param data the data to use
     */
    public RandomCode(int numClasses, int numCodes, Instances data) {
      r = data.getRandomNumberGenerator(m_Seed);
      numCodes = Math.max(2, numCodes); // Need at least two classes
      m_Codebits = new boolean[numCodes][numClasses];
      int i = 0;
      do {
        randomize();
        //System.err.println(this);
      } while (!good() && (i++ < 100));
      //System.err.println("Code:\n" + this);
    }

    private boolean good() {
      boolean [] ninClass = new boolean[m_Codebits[0].length];
      boolean [] ainClass = new boolean[m_Codebits[0].length];
      for (int i = 0; i < ainClass.length; i++) {
	ainClass[i] = true;
      }

      for (int i = 0; i < m_Codebits.length; i++) {
        boolean ninCode = false;
        boolean ainCode = true;
        for (int j = 0; j < m_Codebits[i].length; j++) {
          boolean current = m_Codebits[i][j];
          ninCode = ninCode || current;
          ainCode = ainCode && current;
          ninClass[j] = ninClass[j] || current;
          ainClass[j] = ainClass[j] && current;
        }
        if (!ninCode || ainCode) {
          return false;
        }
      }
      for (int j = 0; j < ninClass.length; j++) {
        if (!ninClass[j] || ainClass[j]) {
          return false;
        }
      }
      return true;
    }

    /**
     * randomizes
     */
    private void randomize() {
      for (int i = 0; i < m_Codebits.length; i++) {
        for (int j = 0; j < m_Codebits[i].length; j++) {
	  double temp = r.nextDouble();
          m_Codebits[i][j] = (temp < 0.5) ? false : true;
        }
      }
    }
    
    /**
     * Returns the revision string.
     * 
     * @return		the revision
     */
    public String getRevision() {
      return RevisionUtils.extract("$Revision: 1.48 $");
    }
  }

  /*
   * TODO: Constructs codes as per:
   * Bose, R.C., Ray Chaudhuri (1960), On a class of error-correcting
   * binary group codes, Information and Control, 3, 68-79.
   * Hocquenghem, A. (1959) Codes corecteurs d'erreurs, Chiffres, 2, 147-156. 
   */
  //private class BCHCode extends Code {...}

  /** Constructs an exhaustive code assignment */
  private class ExhaustiveCode 
    extends Code {

    /** for serialization */
    static final long serialVersionUID = 8090991039670804047L;
    
    /**
     * constructor
     * 
     * @param numClasses the number of classes
     */
    public ExhaustiveCode(int numClasses) {
      int width = (int)Math.pow(2, numClasses - 1) - 1;
      m_Codebits = new boolean[width][numClasses];
      for (int j = 0; j < width; j++) {
        m_Codebits[j][0] = true;
      }
      for (int i = 1; i < numClasses; i++) {
        int skip = (int) Math.pow(2, numClasses - (i + 1));
        for(int j = 0; j < width; j++) {
          m_Codebits[j][i] = ((j / skip) % 2 != 0);
        }
      }
      //System.err.println("Code:\n" + this);
    }
    
    /**
     * Returns the revision string.
     * 
     * @return		the revision
     */
    public String getRevision() {
      return RevisionUtils.extract("$Revision: 1.48 $");
    }
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    // class
    result.disableAllClasses();
    result.disableAllClassDependencies();
    result.enable(Capability.NOMINAL_CLASS);
    
    return result;
  }

  /**
   * Builds the classifiers.
   *
   * @param insts the training data.
   * @throws Exception if a classifier can't be built
   */
  public void buildClassifier(Instances insts) throws Exception {

    Instances newInsts;

    // can classifier handle the data?
    getCapabilities().testWithFail(insts);

    // remove instances with missing class
    insts = new Instances(insts);
    insts.deleteWithMissingClass();
    
    if (m_Classifier == null) 
    {
      throw new Exception("No base classifier has been set!");
    }
    m_ZeroR = new ZeroR();
    m_ZeroR.buildClassifier(insts);

    m_TwoClassDataset = null;

    int numClassifiers = insts.numClasses();
    if (numClassifiers <= 2) 
    {

      m_Classifiers = Classifier.makeCopies(m_Classifier, 1);
      m_Classifiers[0].buildClassifier(insts);

      m_ClassFilters = null;

    }
    else if (m_Method == METHOD_1_AGAINST_1) 
    {
      // generate fastvector of pairs
      FastVector pairs = new FastVector();
      for (int i=0; i<insts.numClasses(); i++)
      {
      	for (int j=0; j<insts.numClasses(); j++) 
      	{
      		if (j<=i) continue;
      		int[] pair = new int[2];
      		pair[0] = i; pair[1] = j;
      		pairs.addElement(pair);
      	}
      }

      numClassifiers = pairs.size();
      m_Classifiers = Classifier.makeCopies(m_Classifier, numClassifiers);
      m_ClassFilters = new Filter[numClassifiers];
      m_SumOfWeights = new double[numClassifiers];

      // generate the classifiers
      for (int i=0; i<numClassifiers; i++) 
      {
      	RemoveWithValues classFilter = new RemoveWithValues();
      	classFilter.setAttributeIndex("" + (insts.classIndex() + 1));
      	classFilter.setModifyHeader(true);
      	classFilter.setInvertSelection(true);
      	classFilter.setNominalIndicesArr((int[])pairs.elementAt(i));
      	Instances tempInstances = new Instances(insts, 0);
      	tempInstances.setClassIndex(-1);
      	classFilter.setInputFormat(tempInstances);
      	newInsts = Filter.useFilter(insts, classFilter);
      	if (newInsts.numInstances() > 0) {
      		newInsts.setClassIndex(insts.classIndex());
      		m_Classifiers[i].buildClassifier(newInsts);
      		m_ClassFilters[i] = classFilter;
      		m_SumOfWeights[i] = newInsts.sumOfWeights();
      	}
      	else 
      	{
      		m_Classifiers[i] = null;
      		m_ClassFilters[i] = null;
      	}
      }

      // construct a two-class header version of the dataset
      m_TwoClassDataset = new Instances(insts, 0);
      int classIndex = m_TwoClassDataset.classIndex();
      m_TwoClassDataset.setClassIndex(-1);
      m_TwoClassDataset.deleteAttributeAt(classIndex);
      FastVector classLabels = new FastVector();
      classLabels.addElement("class0");
      classLabels.addElement("class1");
      m_TwoClassDataset.insertAttributeAt(new Attribute("class", classLabels),
					  classIndex);
      m_TwoClassDataset.setClassIndex(classIndex);

    } 
    else 
    {
   	 // use error correcting code style methods
   	Code code = null;
      switch (m_Method) {
      case METHOD_ERROR_EXHAUSTIVE:
        code = new ExhaustiveCode(numClassifiers);
        break;
      case METHOD_ERROR_RANDOM:
        code = new RandomCode(numClassifiers, 
                              (int)(numClassifiers * m_RandomWidthFactor),
			      insts);
        break;
      case METHOD_1_AGAINST_ALL:
        code = new StandardCode(numClassifiers);
        break;
      default:
        throw new Exception("Unrecognized correction code type");
      }
      numClassifiers = code.size();
      m_Classifiers = Classifier.makeCopies(m_Classifier, numClassifiers);
      m_ClassFilters = new MakeIndicator[numClassifiers];
      for (int i = 0; i < m_Classifiers.length; i++) 
      {
      	m_ClassFilters[i] = new MakeIndicator();
      	MakeIndicator classFilter = (MakeIndicator) m_ClassFilters[i];
      	classFilter.setAttributeIndex("" + (insts.classIndex() + 1));
      	classFilter.setValueIndices(code.getIndices(i));
      	classFilter.setNumeric(false);
      	classFilter.setInputFormat(insts);
      	newInsts = Filter.useFilter(insts, m_ClassFilters[i]);
      	m_Classifiers[i].buildClassifier(newInsts);
      }
    }
    m_ClassAttribute = insts.classAttribute();
  }

  /**
   * Returns the individual predictions of the base classifiers
   * for an instance. Used by StackedMultiClassClassifier.
   * Returns the probability for the second "class" predicted
   * by each base classifier.
   *
   * @param inst the instance to get the prediction for
   * @return the individual predictions
   * @throws Exception if the predictions can't be computed successfully
   */
  public double[] individualPredictions(Instance inst) throws Exception {
    
    double[] result = null;

    if (m_Classifiers.length == 1) {
      result = new double[1];
      result[0] = m_Classifiers[0].distributionForInstance(inst)[1];
    } else {
      result = new double[m_ClassFilters.length];
      for(int i = 0; i < m_ClassFilters.length; i++) {
	if (m_Classifiers[i] != null) {
	  if (m_Method == METHOD_1_AGAINST_1) {    
	    Instance tempInst = (Instance)inst.copy(); 
	    tempInst.setDataset(m_TwoClassDataset);
	    result[i] = m_Classifiers[i].distributionForInstance(tempInst)[1];  
	  } else {
	    m_ClassFilters[i].input(inst);
	    m_ClassFilters[i].batchFinished();
	    result[i] = m_Classifiers[i].
	      distributionForInstance(m_ClassFilters[i].output())[1];
	  }
	}
      }
    }
    return result;
  }

  /**
   * Returns the distribution for an instance.
   *
   * @param inst the instance to get the distribution for
   * @return the distribution
   * @throws Exception if the distribution can't be computed successfully
   */
  public double[] distributionForInstance(Instance inst) throws Exception {
    
    if (m_Classifiers.length == 1) 
    {
      return m_Classifiers[0].distributionForInstance(inst);
    }
    
    double[] probs = new double[inst.numClasses()];

    if (m_Method == METHOD_1_AGAINST_1)
    {
      double[][] r = new double[inst.numClasses()][inst.numClasses()];
      double[][] n = new double[inst.numClasses()][inst.numClasses()];

      for(int i = 0; i < m_ClassFilters.length; i++) 
      {
      	if (m_Classifiers[i] != null) {
      		Instance tempInst = (Instance)inst.copy(); 
      		tempInst.setDataset(m_TwoClassDataset);
      		double [] current = m_Classifiers[i].distributionForInstance(tempInst);  
      		Range range = new Range(((RemoveWithValues)m_ClassFilters[i])
				  .getNominalIndices());
      		range.setUpper(m_ClassAttribute.numValues());
      		int[] pair = range.getSelection();
      		if (m_pairwiseCoupling && inst.numClasses() > 2) {
      			r[pair[0]][pair[1]] = current[0];
      			n[pair[0]][pair[1]] = m_SumOfWeights[i];
      		}
      		else {
      			if (current[0] > current[1]) {
      				probs[pair[0]] += 1.0;
      			}
      			else {
      				probs[pair[1]] += 1.0;
      			}
      		}
      	}
      }
      if (m_pairwiseCoupling && inst.numClasses() > 2) {
        return pairwiseCoupling(n, r);
      }
    } 
    else 
    {
      // error correcting style methods
      for(int i = 0; i < m_ClassFilters.length; i++) 	
      {
      	m_ClassFilters[i].input(inst);
      	m_ClassFilters[i].batchFinished();
      	double [] current = m_Classifiers[i].distributionForInstance(m_ClassFilters[i].output());
      	//Calibrate the binary classifier scores
      	
      	
      	for (int j = 0; j < m_ClassAttribute.numValues(); j++)
      	{
      		if (((MakeIndicator)m_ClassFilters[i]).getValueRange().isInRange(j))
      		{
      			probs[j] += current[1];
      		}
      		else 
      		{
      			probs[j] += current[0];
      		}
      	}
      }
    }
    
    if (Utils.gr(Utils.sum(probs), 0)) 
    {
      Utils.normalize(probs);
      return probs;
    }
    else {
   	 return m_ZeroR.distributionForInstance(inst);
    }
  }

  public double[][] calibratedDistributionForTestInstances(Instances test)
  throws Exception
  {
	  double [][] binProbs = new double[m_Classifiers.length][test.numInstances()];
	  double [][] calibratedProbs = new double[m_Classifiers.length][test.numInstances()];
	  boolean [] target = new boolean[test.numInstances()];
	  int prior1=0;
	  int prior0=0;
	  if (m_Classifiers.length == 1) 
	  {
	  	  for (int i = 0; i < test.numInstances(); i++)
	  	  {
	      		Instance inst = test.instance(i);
	      		//m_ClassFilters[0].input(inst);
	      		//m_ClassFilters[0].batchFinished();
	      		//Instance filteredInst = m_ClassFilters[i].output();
	  		  
	  		  
	  		  
	  		  //binProbs[0][i] = (200*m_Classifiers[0].distributionForInstance(inst)[1])-100;
	        binProbs[0][i] = m_Classifiers[0].distributionForInstance(inst)[1];
	  		  if(target[i] = inst.classValue() == 1.0)
	  			  prior1++;
	  		  else
	  			  prior0++;
	  	  }
	  	  calibratedProbs[0] =  sigTraining(binProbs[0], target, prior1, prior0);
	  	  return calibratedProbs;
	  }
	  else
	  {
	    
	    double[] probs = new double[test.classAttribute().numValues()];

	    if (m_Method == METHOD_1_AGAINST_1)
	    {
	   	 throw new Exception("Not implemented for Method 1 against 1");
	      /*double[][] r = new double[inst.numClasses()][inst.numClasses()];
	      double[][] n = new double[inst.numClasses()][inst.numClasses()];

	      for(int i = 0; i < m_ClassFilters.length; i++) 
	      {
	      	if (m_Classifiers[i] != null) {
	      		Instance tempInst = (Instance)inst.copy(); 
	      		tempInst.setDataset(m_TwoClassDataset);
	      		double [] current = m_Classifiers[i].distributionForInstance(tempInst);  
	      		Range range = new Range(((RemoveWithValues)m_ClassFilters[i])
					  .getNominalIndices());
	      		range.setUpper(m_ClassAttribute.numValues());
	      		int[] pair = range.getSelection();
	      		if (m_pairwiseCoupling && inst.numClasses() > 2) {
	      			r[pair[0]][pair[1]] = current[0];
	      			n[pair[0]][pair[1]] = m_SumOfWeights[i];
	      		}
	      		else {
	      			if (current[0] > current[1]) {
	      				probs[pair[0]] += 1.0;
	      			}
	      			else {
	      				probs[pair[1]] += 1.0;
	      			}
	      		}
	      	}
	      }
	      if (m_pairwiseCoupling && inst.numClasses() > 2) {
	        return pairwiseCoupling(n, r);
	      }*/
	    } 
	    else 
	    {
	      // error correcting style methods
		   for(int i = 0; i < m_ClassFilters.length; i++) 
		   {
		   	prior1=0;
		   	prior0=0;
		   	for (int k = 0; k < test.numInstances(); k++)	
	      	{
	      		Instance inst = test.instance(k);
	      		m_ClassFilters[i].input(inst);
	      		m_ClassFilters[i].batchFinished();
	      		Instance filteredInst = m_ClassFilters[i].output();
	      		//binProbs[i][k] = (200*m_Classifiers[i].distributionForInstance(filteredInst)[1]) - 100;
	      		binProbs[i][k] = m_Classifiers[i].distributionForInstance(filteredInst)[1];

	 	  		   //System.out.println(binProbs[i][k] + " " + inst.classValue());
	      		//System.out.println("Class value: " + filteredInst.classValue() + " " + filteredInst.stringValue(filteredInst.numAttributes()-1) + " " + m_Classifiers[i].distributionForInstance(filteredInst)[0] + " " + m_Classifiers[i].distributionForInstance(filteredInst)[1]);
	      		if(target[k]=(filteredInst.classValue() == 1.0))
		  			  prior1++;
	      		else
		  			  prior0++;
	      	
	      	/*for (int j = 0; j < m_ClassAttribute.numValues(); j++)
	      	{
	      		if (((MakeIndicator)m_ClassFilters[i]).getValueRange().isInRange(j))
	      		{
	      			binProbs[j] += current[1];
	      		}
	      		else 
	      		{
	      			binProbs[j] += current[0];
	      		}
	      	}*/
	      	}
	   		calibratedProbs[i] = sigTraining(binProbs[i],target, prior1, prior0);
		   }
/*	   	for (int k = 0; k < test.numInstances(); k++)	
      	{
			   for (int i =0; i < 3; i++)
	   		System.out.println(i + " " + k + " cal: " + calibratedProbs[i][k] + " " + binProbs[i][k]);
      	}
	*/    }
	  }
	  for (int i = 0; i < test.numInstances(); i++)
	  {
		  double sum = 0;
		  for (int j = 0; j < m_Classifiers.length; j++)
		  {
			  sum+=calibratedProbs[j][i];
		  }
		  for (int j = 0; j < m_Classifiers.length; j++)
			  calibratedProbs[j][i]/=sum;
	  }
	  return calibratedProbs;
	    /*
	    if (Utils.gr(Utils.sum(probs), 0)) 
	    {
	      Utils.normalize(probs);
	      return probs;
	    }
	    else {
	   	 return m_ZeroR.distributionForInstance(inst);
	    }*/
  }

  private double[] sigTraining(double[] out, boolean[] target, int prior1, int prior0) 
  {
	  double A = 0;
	  double B = Math.log((prior0 + 1.0)/(prior1 + 1.0));
	  double hiTarget = (prior1+1)/(prior1+2);
	  double loTarget = 1.0/(prior0+2);
	  double lambda = 0.001;
	  double olderr = Double.MAX_VALUE;
	  double[] pp = new double[out.length];
	  
	  for (int i = 0; i < out.length; i++)
	  {
		  pp[i]=(prior1+1)/(prior0+prior1+2);
	  }
	  
	  int count = 0;
	  for (int it =0; it < 100; it++)
	  {
		  double a=0;
		  double b=0;
		  double c=0;
		  double d=0;
		  double e=0;
		  double t=0;
		  
		  for (int i=0; i < out.length; i++)
		  {
			  if (target[i])
				  t = hiTarget;
			  else
				  t = loTarget;
			  double d1=pp[i]-t;
			  double d2 = pp[i]*(1-pp[i]);
			  a += out[i]*out[i]*d2;
			  b += d2;
			  c += out[i]*d2;
			  d += out[i] *d1;
			  e += d1;
		  }
		  
		  if (Math.abs(d) < 0.000000001 && Math.abs(e) < 0.000000001)
			  break;
		  double oldA = A;
		  double oldB = B;
		  double err = 0;
		  
		  while(true)
		  {
			  double det = (a+lambda) * (b+lambda) - (c*c);
			  if (det == 0)
			  {
				  lambda *= 10;
				  continue;
			  }
			  A = oldA + ((b+lambda)*d-(c*e))/det;
			  B = oldB + ((a+lambda)*e-(c*d))/det;
			  
			  err=0;
			  for (int i = 0; i < out.length; i++)
			  {
				  double p = 1.0/(1.0+Math.exp(-1.0 * ((out[i]*A)+B)));
				  pp[i] = p;
/*				  if (p==0)
					  err -= t * -200.0;
				  else if(p==1)
					  err -= (1.0-t) * -200.0;
				  else*/
					  err -= (t*Math.log(p)) + ((1.0-t)*Math.log(1.0-p));					  
					  //err -= (t*-200.0) + ((1.0-t)*Math.log(1.0-p));
				  //else
			  }
			  if (err < olderr*(1.0000001))
			  {
				  lambda *= 0.1;
				  break;
			  }
			  
			  lambda *= 10;
			  
			  if (lambda >= 1000000)
				  break;
		  }
		  double diff = err-olderr;
		  double scale = 0.5*(err+olderr+1);
		  if (diff > -0.001 * scale && diff < 0.0000001 * scale)
			  count++;
		  else
			  count = 0;
		  olderr=err;
		  if (count == 3)
			  break;
		  
	  }
	  //Calibrate the scores into probabilities
	  double [] calScores = new double[out.length];
	  for (int i = 0; i < out.length; i++)
	  {
		  
		  calScores[i] = 1.0/(1.0 + Math.exp(-1.0 * ((A * out[i]) + B)));
	  }
	  return calScores;
	  
  }

/**
   * Prints the classifiers.
   * 
   * @return a string representation of the classifier
   */
  public String toString() {

    if (m_Classifiers == null) {
      return "MultiClassClassifier: No model built yet.";
    }
    StringBuffer text = new StringBuffer();
    text.append("MultiClassClassifier\n\n");
    for (int i = 0; i < m_Classifiers.length; i++) {
      text.append("Classifier ").append(i + 1);
      if (m_Classifiers[i] != null) {
        if ((m_ClassFilters != null) && (m_ClassFilters[i] != null)) {
	  if (m_ClassFilters[i] instanceof RemoveWithValues) {
	    Range range = new Range(((RemoveWithValues)m_ClassFilters[i])
				    .getNominalIndices());
	    range.setUpper(m_ClassAttribute.numValues());
	    int[] pair = range.getSelection();
	    text.append(", " + (pair[0]+1) + " vs " + (pair[1]+1));
	  } else if (m_ClassFilters[i] instanceof MakeIndicator) {
	    text.append(", using indicator values: ");
	    text.append(((MakeIndicator)m_ClassFilters[i]).getValueRange());
	  }
        }
        text.append('\n');
        text.append(m_Classifiers[i].toString() + "\n\n");
      } else {
        text.append(" Skipped (no training examples)\n");
      }
    }

    return text.toString();
  }

  /**
   * Returns an enumeration describing the available options
   *
   * @return an enumeration of all the available options
   */
  public Enumeration listOptions()  {

    Vector vec = new Vector(4);
    
    vec.addElement(new Option(
       "\tSets the method to use. Valid values are 0 (1-against-all),\n"
       +"\t1 (random codes), 2 (exhaustive code), and 3 (1-against-1). (default 0)\n",
       "M", 1, "-M <num>"));
    vec.addElement(new Option(
       "\tSets the multiplier when using random codes. (default 2.0)",
       "R", 1, "-R <num>"));
    vec.addElement(new Option(
        "\tUse pairwise coupling (only has an effect for 1-against1)",
        "P", 0, "-P"));

    Enumeration enu = super.listOptions();
    while (enu.hasMoreElements()) {
      vec.addElement(enu.nextElement());
    }
    return vec.elements();
  }

  /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -M &lt;num&gt;
   *  Sets the method to use. Valid values are 0 (1-against-all),
   *  1 (random codes), 2 (exhaustive code), and 3 (1-against-1). (default 0)
   * </pre>
   * 
   * <pre> -R &lt;num&gt;
   *  Sets the multiplier when using random codes. (default 2.0)</pre>
   * 
   * <pre> -P
   *  Use pairwise coupling (only has an effect for 1-against1)</pre>
   * 
   * <pre> -S &lt;num&gt;
   *  Random number seed.
   *  (default 1)</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   * <pre> -W
   *  Full name of base classifier.
   *  (default: weka.classifiers.functions.Logistic)</pre>
   * 
   * <pre> 
   * Options specific to classifier weka.classifiers.functions.Logistic:
   * </pre>
   * 
   * <pre> -D
   *  Turn on debugging output.</pre>
   * 
   * <pre> -R &lt;ridge&gt;
   *  Set the ridge in the log-likelihood.</pre>
   * 
   * <pre> -M &lt;number&gt;
   *  Set the maximum number of iterations (default -1, until convergence).</pre>
   * 
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
  
    String errorString = Utils.getOption('M', options);
    if (errorString.length() != 0) {
      setMethod(new SelectedTag(Integer.parseInt(errorString), 
                                             TAGS_METHOD));
    } else {
      setMethod(new SelectedTag(METHOD_1_AGAINST_ALL, TAGS_METHOD));
    }

    String rfactorString = Utils.getOption('R', options);
    if (rfactorString.length() != 0) {
      setRandomWidthFactor((new Double(rfactorString)).doubleValue());
    } else {
      setRandomWidthFactor(2.0);
    }

    setUsePairwiseCoupling(Utils.getFlag('P', options));

    super.setOptions(options);
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {

    String [] superOptions = super.getOptions();
    String [] options = new String [superOptions.length + 5];

    int current = 0;


    options[current++] = "-M";
    options[current++] = "" + m_Method;

    if (getUsePairwiseCoupling()) {
      options[current++] = "-P";
    }
    
    options[current++] = "-R";
    options[current++] = "" + m_RandomWidthFactor;

    System.arraycopy(superOptions, 0, options, current, 
		     superOptions.length);

    current += superOptions.length;
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }

  /**
   * @return a description of the classifier suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {

    return "A metaclassifier for handling multi-class datasets with 2-class "
      + "classifiers. This classifier is also capable of "
      + "applying error correcting output codes for increased accuracy.";
  }

  /**
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String randomWidthFactorTipText() {

    return "Sets the width multiplier when using random codes. The number "
      + "of codes generated will be thus number multiplied by the number of "
      + "classes.";
  }

  /**
   * Gets the multiplier when generating random codes. Will generate
   * numClasses * m_RandomWidthFactor codes.
   *
   * @return the width multiplier
   */
  public double getRandomWidthFactor() {

    return m_RandomWidthFactor;
  }
  
  /**
   * Sets the multiplier when generating random codes. Will generate
   * numClasses * m_RandomWidthFactor codes.
   *
   * @param newRandomWidthFactor the new width multiplier
   */
  public void setRandomWidthFactor(double newRandomWidthFactor) {

    m_RandomWidthFactor = newRandomWidthFactor;
  }
  
  /**
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String methodTipText() {
    return "Sets the method to use for transforming the multi-class problem into "
      + "several 2-class ones."; 
  }

  /**
   * Gets the method used. Will be one of METHOD_1_AGAINST_ALL,
   * METHOD_ERROR_RANDOM, METHOD_ERROR_EXHAUSTIVE, or METHOD_1_AGAINST_1.
   *
   * @return the current method.
   */
  public SelectedTag getMethod() {
      
    return new SelectedTag(m_Method, TAGS_METHOD);
  }

  /**
   * Sets the method used. Will be one of METHOD_1_AGAINST_ALL,
   * METHOD_ERROR_RANDOM, METHOD_ERROR_EXHAUSTIVE, or METHOD_1_AGAINST_1.
   *
   * @param newMethod the new method.
   */
  public void setMethod(SelectedTag newMethod) {
    
    if (newMethod.getTags() == TAGS_METHOD) {
      m_Method = newMethod.getSelectedTag().getID();
    }
  }

  /**
   * Set whether to use pairwise coupling with 1-vs-1 
   * classification to improve probability estimates.
   *
   * @param p true if pairwise coupling is to be used
   */
  public void setUsePairwiseCoupling(boolean p) {
    m_pairwiseCoupling = p;
  }

  /**
   * Gets whether to use pairwise coupling with 1-vs-1 
   * classification to improve probability estimates.
   *
   * @return true if pairwise coupling is to be used
   */
  public boolean getUsePairwiseCoupling() {
    return m_pairwiseCoupling;
  }

  /**
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String usePairwiseCouplingTipText() {
    return "Use pairwise coupling (only has an effect for 1-against-1).";
  }

  /**
   * Implements pairwise coupling.
   *
   * @param n the sum of weights used to train each model
   * @param r the probability estimate from each model
   * @return the coupled estimates
   */
  public static double[] pairwiseCoupling(double[][] n, double[][] r) {

    // Initialize p and u array
    double[] p = new double[r.length];
    for (int i =0; i < p.length; i++) {
      p[i] = 1.0 / (double)p.length;
    }
    double[][] u = new double[r.length][r.length];
    for (int i = 0; i < r.length; i++) {
      for (int j = i + 1; j < r.length; j++) {
	u[i][j] = 0.5;
      }
    }

    // firstSum doesn't change
    double[] firstSum = new double[p.length];
    for (int i = 0; i < p.length; i++) {
      for (int j = i + 1; j < p.length; j++) {
	firstSum[i] += n[i][j] * r[i][j];
	firstSum[j] += n[i][j] * (1 - r[i][j]);
      }
    }

    // Iterate until convergence
    boolean changed;
    do {
      changed = false;
      double[] secondSum = new double[p.length];
      for (int i = 0; i < p.length; i++) {
	for (int j = i + 1; j < p.length; j++) {
	  secondSum[i] += n[i][j] * u[i][j];
	  secondSum[j] += n[i][j] * (1 - u[i][j]);
	}
      }
      for (int i = 0; i < p.length; i++) {
	if ((firstSum[i] == 0) || (secondSum[i] == 0)) {
	  if (p[i] > 0) {
	    changed = true;
	  }
	  p[i] = 0;
	} else {
	  double factor = firstSum[i] / secondSum[i];
	  double pOld = p[i];
	  p[i] *= factor;
	  if (Math.abs(pOld - p[i]) > 1.0e-3) {
	    changed = true;
	  }
	}
      }
      Utils.normalize(p);
      for (int i = 0; i < r.length; i++) {
	for (int j = i + 1; j < r.length; j++) {
	  u[i][j] = p[i] / (p[i] + p[j]);
	}
      }
    } while (changed);
    return p;
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 1.48 $");
  }

  /**
   * Main method for testing this class.
   *
   * @param argv the options
   */
  public static void main(String [] argv) {
    runClassifier(new MultiClassClassifier(), argv);
  }
}

