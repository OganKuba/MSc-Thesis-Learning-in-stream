package thesis.selection;

public interface FilterRanker {

    void update(int[] discretizedInstance, int classLabel);

    double[] getFeatureScores();

    int[] selectTopK(int k);

    void reset();

    int getNumFeatures();

    default String name() { return getClass().getSimpleName(); }
}