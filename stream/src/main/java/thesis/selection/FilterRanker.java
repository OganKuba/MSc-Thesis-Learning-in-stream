package thesis.selection;

public interface FilterRanker {

    void update(int[] discretizedInstance, int classLabel);

    double[] getFeatureScores();

    int[] selectTopK(int k);

    int[] selectTopK(int k, int[] preferredOrder);

    void reset();

    void resetFeature(int featureIdx);

    void decay(double factor);

    void decayFeature(int featureIdx, double factor);

    boolean isReady();

    long getTotalSamples();

    long getRejectedSamples();

    int getNumFeatures();

    default String name() { return getClass().getSimpleName(); }

    default int[] selectTopK(int k, int[] preferredOrder, double tieEpsilon) {
        return selectTopK(k, preferredOrder);
    }
}