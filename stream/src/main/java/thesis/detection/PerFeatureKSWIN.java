package thesis.detection;

import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;

public class PerFeatureKSWIN {

    private final int numFeatures;
    private final double alpha;
    private final double fdrQ;
    private final int windowSize;
    private final KSWINSingleFeature[] detectors;
    private final double[] lastPValues;

    public PerFeatureKSWIN(int numFeatures, double alpha, int windowSize) {
        this(numFeatures, alpha, windowSize, 0.05);
    }

    public PerFeatureKSWIN(int numFeatures, double alpha, int windowSize, double fdrQ) {
        if (numFeatures < 1) {
            throw new IllegalArgumentException("numFeatures must be >= 1");
        }
        if (!(fdrQ > 0.0 && fdrQ < 1.0)) {
            throw new IllegalArgumentException("fdrQ must be in (0,1), got " + fdrQ);
        }
        this.numFeatures = numFeatures;
        this.alpha = alpha;
        this.windowSize = windowSize;
        this.fdrQ = fdrQ;
        this.detectors = new KSWINSingleFeature[numFeatures];
        this.lastPValues = new double[numFeatures];
        Arrays.fill(lastPValues, 1.0);
        for (int i = 0; i < numFeatures; i++) {
            detectors[i] = new KSWINSingleFeature(windowSize, alpha);
        }
    }

    public void update(double[] featureValues) {
        if (featureValues.length != numFeatures) {
            throw new IllegalArgumentException(
                    "expected " + numFeatures + " features, got " + featureValues.length);
        }
        for (int i = 0; i < numFeatures; i++) {
            detectors[i].update(featureValues[i]);
        }
    }

    public Set<Integer> getDriftingFeatures() {
        for (int i = 0; i < numFeatures; i++) {
            detectors[i].testDrift();
            lastPValues[i] = detectors[i].getPValue();
        }

        Integer[] order = new Integer[numFeatures];
        for (int i = 0; i < numFeatures; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(lastPValues[a], lastPValues[b]));

        int cutoff = -1;
        for (int rank = 1; rank <= numFeatures; rank++) {
            int idx = order[rank - 1];
            double threshold = (rank / (double) numFeatures) * fdrQ;
            if (lastPValues[idx] <= threshold) {
                cutoff = rank;
            }
        }

        Set<Integer> flagged = new LinkedHashSet<>();
        if (cutoff > 0) {
            for (int rank = 1; rank <= cutoff; rank++) {
                flagged.add(order[rank - 1]);
            }
        }
        return flagged;
    }

    public Set<Integer> getRawDriftingFeatures() {
        Set<Integer> flagged = new HashSet<>();
        for (int i = 0; i < numFeatures; i++) {
            if (detectors[i].testDrift()) flagged.add(i);
            lastPValues[i] = detectors[i].getPValue();
        }
        return flagged;
    }

    public double[] getLastPValues() {
        return Arrays.copyOf(lastPValues, lastPValues.length);
    }

    public double getKSStatistic(int featureIdx) {
        return detectors[featureIdx].getKSStatistic();
    }

    public boolean isReady() {
        for (KSWINSingleFeature d : detectors) {
            if (!d.isReady()) return false;
        }
        return true;
    }

    public void resetAll() {
        for (KSWINSingleFeature d : detectors) d.reset();
        Arrays.fill(lastPValues, 1.0);
    }

    public void resetFeature(int idx) {
        detectors[idx].reset();
        lastPValues[idx] = 1.0;
    }

    public int getNumFeatures() { return numFeatures; }
    public double getAlpha()    { return alpha; }
    public double getFdrQ()     { return fdrQ; }
    public int getWindowSize()  { return windowSize; }
}