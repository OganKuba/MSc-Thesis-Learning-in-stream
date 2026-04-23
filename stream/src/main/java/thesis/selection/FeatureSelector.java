package thesis.selection;

import java.util.Set;

public interface FeatureSelector {

    void initialize(double[][] initialWindow, int[] labels);

    void update(double[] instance, int classLabel,
                boolean driftAlarm, Set<Integer> driftingFeatures);

    int[] getSelectedFeatures();

    int[] getCurrentSelection();

    double[] filterInstance(double[] fullInstance);

    int getNumFeatures();

    int getK();

    boolean isInitialized();

    default String name() { return getClass().getSimpleName(); }
}