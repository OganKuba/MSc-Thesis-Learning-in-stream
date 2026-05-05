package thesis.selection;

import lombok.Getter;

import java.util.Arrays;

@Getter
public abstract class AbstractFrequencyRanker implements FilterRanker {

    public static final int UNKNOWN_BIN = -1;

    protected final int numFeatures;
    protected final int numBins;
    protected final int numClasses;
    protected final int minSamplesReady;

    protected final double[][][] joint;
    protected final double[][] featureBinTotals;
    protected final double[] featureTotals;
    protected final double[][] featureClassMarginals;

    protected long totalSamples;
    protected long rejectedSamples;

    protected AbstractFrequencyRanker(int numFeatures, int numBins, int numClasses) {
        this(numFeatures, numBins, numClasses, 50);
    }

    protected AbstractFrequencyRanker(int numFeatures, int numBins, int numClasses, int minSamplesReady) {
        if (numFeatures < 1) throw new IllegalArgumentException("numFeatures must be >= 1");
        if (numBins < 2) throw new IllegalArgumentException("numBins must be >= 2");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        if (minSamplesReady < 1) throw new IllegalArgumentException("minSamplesReady must be >= 1");
        this.numFeatures = numFeatures;
        this.numBins = numBins;
        this.numClasses = numClasses;
        this.minSamplesReady = minSamplesReady;
        this.joint = new double[numFeatures][numBins][numClasses];
        this.featureBinTotals = new double[numFeatures][numBins];
        this.featureTotals = new double[numFeatures];
        this.featureClassMarginals = new double[numFeatures][numClasses];
    }

    @Override
    public void update(int[] discretizedInstance, int classLabel) {
        if (discretizedInstance.length != numFeatures) {
            throw new IllegalArgumentException("expected " + numFeatures + " features, got " + discretizedInstance.length);
        }
        if (classLabel < 0 || classLabel >= numClasses) {
            rejectedSamples++;
            return;
        }
        boolean usedAny = false;
        for (int f = 0; f < numFeatures; f++) {
            int b = discretizedInstance[f];
            if (b == UNKNOWN_BIN) continue;
            if (b < 0 || b >= numBins) continue;
            joint[f][b][classLabel] += 1.0;
            featureBinTotals[f][b] += 1.0;
            featureTotals[f] += 1.0;
            featureClassMarginals[f][classLabel] += 1.0;
            usedAny = true;
        }
        if (usedAny) totalSamples++;
        else rejectedSamples++;
    }

    @Override
    public double[] getFeatureScores() {
        double[] scores = new double[numFeatures];
        for (int f = 0; f < numFeatures; f++) scores[f] = score(f);
        return scores;
    }

    protected abstract double score(int featureIdx);

    private double safeScore(double score) {
        return (Double.isFinite(score) && score >= 0.0) ? score : 0.0;
    }

    @Override
    public int[] selectTopK(int k) {
        return selectTopK(k, null);
    }

    @Override
    public int[] selectTopK(int k, int[] preferredOrder) {
        if (k < 1 || k > numFeatures) {
            throw new IllegalArgumentException("k must be in [1, " + numFeatures + "]");
        }

        double[] scores = getFeatureScores();
        double[] safeScores = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            safeScores[i] = safeScore(scores[i]);
        }

        int[] preferRank = new int[numFeatures];
        Arrays.fill(preferRank, Integer.MAX_VALUE);

        if (preferredOrder != null) {
            for (int r = 0; r < preferredOrder.length; r++) {
                int idx = preferredOrder[r];
                if (idx >= 0 && idx < numFeatures && preferRank[idx] == Integer.MAX_VALUE) {
                    preferRank[idx] = r;
                }
            }
        }

        Integer[] idx = new Integer[numFeatures];
        for (int i = 0; i < numFeatures; i++) idx[i] = i;

        Arrays.sort(idx, (a, b) -> {
            int cmp = Double.compare(safeScores[b], safeScores[a]); // descending
            if (cmp != 0) return cmp;

            cmp = Integer.compare(preferRank[a], preferRank[b]);
            if (cmp != 0) return cmp;

            return Integer.compare(a, b);
        });

        int[] out = new int[k];
        for (int i = 0; i < k; i++) out[i] = idx[i];

        return out;
    }

    @Override
    public void reset() {
        for (int f = 0; f < numFeatures; f++) resetFeature(f);
        totalSamples = 0;
        rejectedSamples = 0;
    }

    @Override
    public void resetFeature(int featureIdx) {
        for (int b = 0; b < numBins; b++) Arrays.fill(joint[featureIdx][b], 0.0);
        Arrays.fill(featureBinTotals[featureIdx], 0.0);
        featureTotals[featureIdx] = 0.0;
        Arrays.fill(featureClassMarginals[featureIdx], 0.0);
    }

    @Override
    public void decay(double factor) {
        if (!(factor > 0.0 && factor <= 1.0)) throw new IllegalArgumentException("factor in (0,1]");
        if (factor == 1.0) return;
        for (int f = 0; f < numFeatures; f++) decayFeature(f, factor);
    }

    @Override
    public void decayFeature(int featureIdx, double factor) {
        if (!(factor > 0.0 && factor <= 1.0)) throw new IllegalArgumentException("factor in (0,1]");
        if (factor == 1.0) return;
        for (int b = 0; b < numBins; b++) {
            for (int c = 0; c < numClasses; c++) joint[featureIdx][b][c] *= factor;
            featureBinTotals[featureIdx][b] *= factor;
        }
        for (int c = 0; c < numClasses; c++) featureClassMarginals[featureIdx][c] *= factor;
        featureTotals[featureIdx] *= factor;
    }

    @Override
    public boolean isReady() {
        for (int f = 0; f < numFeatures; f++) {
            if (featureTotals[f] < minSamplesReady) return false;
        }
        return true;
    }

    public int[] selectTopK(int k, int[] preferredOrder, double tieEpsilon) {
        if (k < 1 || k > numFeatures) {
            throw new IllegalArgumentException("k must be in [1, " + numFeatures + "]");
        }
        if (!(tieEpsilon >= 0.0)) {
            throw new IllegalArgumentException("tieEpsilon must be >= 0");
        }

        double[] scores = getFeatureScores();
        double[] safeScores = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            safeScores[i] = safeScore(scores[i]);
        }

        final long[] scoreKey = new long[numFeatures];
        if (tieEpsilon > 0.0) {
            for (int i = 0; i < numFeatures; i++) {
                double v = safeScores[i];
                if (Double.isNaN(v) || Double.isInfinite(v)) {
                    scoreKey[i] = Long.MIN_VALUE;
                } else {
                    scoreKey[i] = Math.round(v / tieEpsilon);
                }
            }
        } else {
            for (int i = 0; i < numFeatures; i++) {
                double v = safeScores[i];
                if (Double.isNaN(v)) {
                    scoreKey[i] = Long.MIN_VALUE;
                } else {
                    scoreKey[i] = Double.doubleToLongBits(v);
                }
            }
        }

        final int[] preferRank = new int[numFeatures];
        Arrays.fill(preferRank, Integer.MAX_VALUE);
        if (preferredOrder != null) {
            for (int r = 0; r < preferredOrder.length; r++) {
                int idx = preferredOrder[r];
                if (idx >= 0 && idx < numFeatures && preferRank[idx] == Integer.MAX_VALUE) {
                    preferRank[idx] = r;
                }
            }
        }

        Integer[] idx = new Integer[numFeatures];
        for (int i = 0; i < numFeatures; i++) idx[i] = i;

        Arrays.sort(idx, (a, b) -> {
            int cmp = Long.compare(scoreKey[b], scoreKey[a]);
            if (cmp != 0) return cmp;
            cmp = Integer.compare(preferRank[a], preferRank[b]);
            if (cmp != 0) return cmp;
            return Integer.compare(a, b);
        });

        int[] out = new int[k];
        for (int i = 0; i < k; i++) out[i] = idx[i];

        return out;
    }


    protected double featureTotal(int featureIdx) { return featureTotals[featureIdx]; }
    protected double[] classMarginal(int featureIdx) { return featureClassMarginals[featureIdx]; }

    @Override public int getNumFeatures() { return numFeatures; }
    @Override public long getTotalSamples() { return totalSamples; }
    @Override public long getRejectedSamples() { return rejectedSamples; }
}