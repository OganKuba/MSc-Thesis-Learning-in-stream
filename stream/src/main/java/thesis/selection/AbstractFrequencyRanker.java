package thesis.selection;

import lombok.Getter;

import java.util.Arrays;
import java.util.Comparator;

@Getter
public abstract class AbstractFrequencyRanker implements FilterRanker {

    protected final int numFeatures;
    protected final int numBins;
    protected final int numClasses;
    protected final int[][][] joint;
    protected final int[][] featureBinTotals;

    protected AbstractFrequencyRanker(int numFeatures, int numBins, int numClasses) {
        if (numFeatures < 1) throw new IllegalArgumentException("numFeatures must be >= 1");
        if (numBins < 2) throw new IllegalArgumentException("numBins must be >= 2");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        this.numFeatures = numFeatures;
        this.numBins = numBins;
        this.numClasses = numClasses;
        this.joint = new int[numFeatures][numBins][numClasses];
        this.featureBinTotals = new int[numFeatures][numBins];
    }

    @Override
    public void update(int[] discretizedInstance, int classLabel) {
        if (discretizedInstance.length != numFeatures) {
            throw new IllegalArgumentException("expected " + numFeatures + " features");
        }
        if (classLabel < 0 || classLabel >= numClasses) {
            throw new IllegalArgumentException("classLabel out of range: " + classLabel);
        }
        for (int f = 0; f < numFeatures; f++) {
            int b = discretizedInstance[f];
            if (b < 0 || b >= numBins) {
                throw new IllegalArgumentException(
                        "bin index out of range for feature " + f + ": " + b);
            }
            joint[f][b][classLabel]++;
            featureBinTotals[f][b]++;
        }
    }

    @Override
    public double[] getFeatureScores() {
        double[] scores = new double[numFeatures];
        for (int f = 0; f < numFeatures; f++) {
            scores[f] = score(f);
        }
        return scores;
    }

    protected abstract double score(int featureIdx);

    @Override
    public int[] selectTopK(int k) {
        if (k < 1 || k > numFeatures) {
            throw new IllegalArgumentException("k must be in [1, " + numFeatures + "]");
        }
        double[] scores = getFeatureScores();
        Integer[] order = new Integer[numFeatures];
        for (int i = 0; i < numFeatures; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingDouble((Integer i) -> -scores[i]));
        int[] out = new int[k];
        for (int i = 0; i < k; i++) out[i] = order[i];
        return out;
    }

    @Override
    public void reset() {
        for (int f = 0; f < numFeatures; f++) resetFeature(f);
    }

    public void resetFeature(int featureIdx) {
        for (int b = 0; b < numBins; b++) {
            Arrays.fill(joint[featureIdx][b], 0);
        }
        Arrays.fill(featureBinTotals[featureIdx], 0);
    }

    protected long featureTotal(int featureIdx) {
        long s = 0;
        for (int b = 0; b < numBins; b++) s += featureBinTotals[featureIdx][b];
        return s;
    }

    protected long[] classMarginal(int featureIdx) {
        long[] m = new long[numClasses];
        for (int b = 0; b < numBins; b++) {
            for (int c = 0; c < numClasses; c++) m[c] += joint[featureIdx][b][c];
        }
        return m;
    }

    @Override public int getNumFeatures() { return numFeatures; }
}