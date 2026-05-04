package thesis.discretization;

import lombok.Getter;

@Getter
public class PiDDiscretizer {

    private final int numFeatures;
    private final int numClasses;
    private final int b1;
    private final int b2;
    private final int warmupN;
    private final int recomputeEvery;
    private final double expandThreshold;
    private final double decayFactor;
    private final FeatureDiscretizer[] features;

    public PiDDiscretizer(int numFeatures, int numClasses) {
        this(numFeatures, numClasses, 64, 8, 500, 1000, 0.20, 1.0);
    }

    public PiDDiscretizer(int numFeatures, int numClasses,
                          int b1, int b2, int warmupN, int recomputeEvery) {
        this(numFeatures, numClasses, b1, b2, warmupN, recomputeEvery, 0.20, 1.0);
    }

    public PiDDiscretizer(int numFeatures, int numClasses,
                          int b1, int b2, int warmupN, int recomputeEvery,
                          double expandThreshold, double decayFactor) {
        if (numFeatures < 1) throw new IllegalArgumentException("numFeatures must be >= 1");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        if (recomputeEvery < 1) throw new IllegalArgumentException("recomputeEvery must be >= 1");
        this.numFeatures = numFeatures;
        this.numClasses = numClasses;
        this.b1 = b1;
        this.b2 = b2;
        this.warmupN = warmupN;
        this.recomputeEvery = recomputeEvery;
        this.expandThreshold = expandThreshold;
        this.decayFactor = decayFactor;
        this.features = new FeatureDiscretizer[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            features[i] = new FeatureDiscretizer(b1, b2, numClasses, warmupN, expandThreshold, decayFactor);
        }
    }

    public void update(double[] featureValues, int classLabel) {
        if (featureValues.length != numFeatures) {
            throw new IllegalArgumentException("expected " + numFeatures + " features");
        }
        if (classLabel < 0 || classLabel >= numClasses) {
            throw new IllegalArgumentException("classLabel out of range: " + classLabel);
        }
        for (int i = 0; i < numFeatures; i++) updateOne(i, featureValues[i], classLabel);
    }

    public void update(int featureIndex, double value, int classLabel) {
        if (featureIndex < 0 || featureIndex >= numFeatures) {
            throw new IndexOutOfBoundsException("featureIndex out of range: " + featureIndex);
        }
        if (classLabel < 0 || classLabel >= numClasses) {
            throw new IllegalArgumentException("classLabel out of range: " + classLabel);
        }
        updateOne(featureIndex, value, classLabel);
    }

    private void updateOne(int featureIndex, double value, int classLabel) {
        FeatureDiscretizer f = features[featureIndex];
        f.update(value, classLabel);
        if (f.isReady() && f.getUpdatesSinceRecompute() >= recomputeEvery) {
            f.recomputeLayer2();
        }
    }

    public int discretize(int featureIndex, double value) {
        return features[featureIndex].discretize(value);
    }

    public int[] discretizeAll(double[] featureValues) {
        if (featureValues.length != numFeatures) {
            throw new IllegalArgumentException("expected " + numFeatures + " features");
        }
        int[] out = new int[numFeatures];
        for (int i = 0; i < numFeatures; i++) out[i] = features[i].discretize(featureValues[i]);
        return out;
    }

    public void recomputeLayer2() { for (FeatureDiscretizer f : features) f.recomputeLayer2(); }
    public void recomputeLayer2(int featureIndex) { features[featureIndex].recomputeLayer2(); }

    public void softResetFeature(int featureIndex, double decay) {
        features[featureIndex].softReset(decay);
    }

    public void onDriftAlarm(java.util.Set<Integer> driftingFeatures, double decay) {
        for (int idx : driftingFeatures) {
            if (idx < 0 || idx >= numFeatures) continue;
            if (features[idx].isReady()) features[idx].softReset(decay);
        }
    }

    public void reset() { for (FeatureDiscretizer f : features) f.reset(); }
    public void resetFeature(int featureIndex) { features[featureIndex].reset(); }

    public boolean isReady(int featureIndex) { return features[featureIndex].isReady(); }
    public boolean isReady() {
        for (FeatureDiscretizer f : features) if (!f.isReady()) return false;
        return true;
    }

    public int[] getL2Counts(int featureIndex) { return features[featureIndex].l2Counts(); }
    public int[][] getL2ClassCounts(int featureIndex) { return features[featureIndex].l2ClassCounts(); }
    public FeatureDiscretizer getFeature(int featureIndex) { return features[featureIndex]; }
}