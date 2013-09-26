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
 *    C45PruneableClassifierTree.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 */

package library;

import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class for handling a tree structure that can
 * be pruned using C4.5 procedures.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 1.15 $
 */

public class MikeC45PruneableClassifierTree extends weka.classifiers.trees.j48.C45PruneableClassifierTree 
{
	
	/** True if the tree is to be pruned. */
	  boolean m_pruneTheTree = false;

	  /** The confidence factor for pruning. */
	  float m_CF = 0.25f;

	  /** Is subtree raising to be performed? */
	  boolean m_subtreeRaising = true;

	  /** Cleanup after the tree has been built. */
	  boolean m_cleanup = true;
	  
	  
	/**
	 * Constructor for pruneable tree structure. Stores reference
	 * to associated training data at each node.
	 *
	 * @param toSelectLocModel selection method for local splitting model
	 * @param pruneTree true if the tree is to be pruned
	 * @param cf the confidence factor for pruning
	 * @param raiseTree
	 * @param cleanup
	 * @throws Exception if something goes wrong
	 */
	public MikeC45PruneableClassifierTree(ModelSelection toSelectLocModel,
			    boolean pruneTree,float cf,
			    boolean raiseTree,
			    boolean cleanup)
	     throws Exception 
   {
		super(toSelectLocModel, pruneTree, cf, raiseTree,cleanup);
		m_pruneTheTree = pruneTree;
	    m_CF = cf;
	    m_subtreeRaising = raiseTree;
	    m_cleanup = cleanup;
		 
	}
	
	  /**
	   * Returns a newly created tree.
	   *
	   * @param data the data to work with
	   * @return the new tree
	   * @throws Exception if something goes wrong
	   */
	  protected ClassifierTree getNewTree(Instances data) throws Exception {
	    
	    MikeC45PruneableClassifierTree newTree = 
	      new MikeC45PruneableClassifierTree(super.m_toSelectModel, m_pruneTheTree, m_CF,
					     m_subtreeRaising, m_cleanup);
	    newTree.buildTree((Instances)data, m_subtreeRaising);

	    return newTree;
	  }
	
	public double[] getDisjunct(Instance instance) throws Exception
   {
   	double [] doubles = new double[instance.numClasses()];

      for (int i = 0; i < doubles.length; i++) 
      {
  	      doubles[i] = getNum(i, instance, 1);
      }
      System.out.print(" " + getDepth(instance, 0));

      return doubles;
    }
	
	/**
	 * Help method for computing class probabilities of 
	 * a given instance.
	 * 
	 * @param classIndex the class index
	 * @param instance the instance to compute the probabilities for
	 * @param weight the weight to use
	 * @return the probs
	 * @throws Exception if something goes wrong
	 */
	private double getNum(int classIndex, Instance instance, double weight) 
	   throws Exception 
   {
	    
	   double prob = 0;
	   
	   if (m_isLeaf) 
	   {
	   	return weight * localModel().distribution().perClass(classIndex);
	   }
	   else
	   {
	      int treeIndex = localModel().whichSubset(instance);
	      if (treeIndex == -1) 
	      {
		      double[] weights = localModel().weights(instance);
		      for (int i = 0; i < m_sons.length; i++) 
		      {
		         if (!son(i).m_isEmpty) 
		         {
		            prob += son(i).getNum(classIndex, instance, 
					   weights[i] * weight);
		         }
		      }
		      return prob;
	      } 
	      else 
	      {
		      if (son(treeIndex).m_isEmpty)
		      {
		      	return weight * localModel().distribution().perClassPerBag(treeIndex, classIndex);
		      }
		      else 
		      {
		         return son(treeIndex).getNum(classIndex, instance, weight);
		      }
	      }
	   }
	}

	/**
	 * Help method for computing the depth of a leaf node for an instance 
	 * 
	 * @param classIndex the class index
	 * @param instance the instance to compute the probabilities for
	 * @return the depth 
	 *@throws Exception if something goes wrong
	 */
	private double getDepth(Instance instance, double d) 
	   throws Exception 
   {
	    
	   double depth = d;
	   
	   if (m_isLeaf) 
	   {
//	   	return d * localModel().distribution().perClass((int) instance.classValue());
	   	return depth;
	   }
	   else
	   {
	      int treeIndex = localModel().whichSubset(instance);
	      if (treeIndex == -1) 
	      {
		      double[] weights = localModel().weights(instance);
		      for (int i = 0; i < m_sons.length; i++) 
		      {
		         if (!son(i).m_isEmpty) 
		         {
		            depth += son(i).getDepth(instance, depth) * weights[i];
		         }
		      }
		      return depth + 1;
	      } 
	      else 
	      {
		      if (son(treeIndex).m_isEmpty)
		      {
		      	return depth + 1;
		      }
		      else 
		      {
		         return son(treeIndex).getDepth(instance, depth) + 1;
		      }
	      }
	   }
	}
	  
	  /**
	    * Method just exists to make program easier to read.
	    */
	   private ClassifierSplitModel localModel() 
	   {
	     
	     return (ClassifierSplitModel)m_localModel;
	   }	
	   
	   /**
	    * Method just exists to make program easier to read.
	    */
	   private MikeC45PruneableClassifierTree son(int index) {
	     return (MikeC45PruneableClassifierTree)m_sons[index];
	   }
  /**
   * Method for building a pruneable classifier tree.
   *
   * @param data the data for building the tree
   * @throws Exception if something goes wrong
   */
  public void buildClassifier(Instances data) throws Exception {

    // can classifier tree handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
   buildTree(data, m_subtreeRaising);
   collapse();
   if (m_pruneTheTree) {
     prune();
   }
   if (m_cleanup) {
     cleanup(new Instances(data, 0));
   }
  }

}
