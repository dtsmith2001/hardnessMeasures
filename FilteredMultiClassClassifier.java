import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;


public class FilteredMultiClassClassifier
extends weka.classifiers.meta.FilteredClassifier
{

	/**
	 * 
	 */
	private static final long serialVersionUID = -2279976963554162490L;

	public double[][] calibratedDistributionForTestInstances(Instances test)
	throws Exception
	{
		//Filter the test instances
		Instances data = Filter.useFilter(test, m_Filter);
		return ((MultiClassClassifier)m_Classifier).calibratedDistributionForTestInstances(data);
		/*if (m_Filter.numPendingOutput() > 0) {
			throw new Exception("Filter output queue not empty!");
		}
		if (!m_Filter.input(instance)) {
			throw new Exception("Filter didn't make the test instance"
			  + " immediately available!");
		}
		m_Filter.batchFinished();
		Instance newInstance = m_Filter.output();
		return m_Classifier.distributionForInstance(newInstance);*/
	}

}
