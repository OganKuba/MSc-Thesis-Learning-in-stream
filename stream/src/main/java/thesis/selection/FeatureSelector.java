package thesis.selection;

import java.util.Set;

public interface FeatureSelector {

    @FunctionalInterface
    interface RankerFactory {
        FilterRanker create(int numFeatures, int numBins, int numClasses);
    }

    void initialize(double[][] initialWindow, int[] labels);

    void update(double[] instance, int classLabel,
                boolean driftAlarm, Set<Integer> driftingFeatures);

    int[] getSelectedFeatures();

    int[] getCurrentSelection();

    double[] filterInstance(double[] fullInstance);

    int getNumFeatures();

    int getK();

    boolean isInitialized();

    default double[] getInitialScores() { return null; }

    /**
     * Why the current selection was last changed: {@code "initial"}, {@code "drift_alarm"},
     * {@code "periodic"}, or {@code "none"} for selectors that never re-select.
     *
     * <p>Needed because the trigger cannot be inferred from the outside. Alarm-driven selectors
     * re-rank only after collecting {@code wPostDrift} instances, so by the time the selection
     * actually changes the alarm flag is long gone — which is why the recorder used to label
     * essentially every S2/S4 re-selection as a generic "selection_change" (4 of 4372 records in
     * E2 carried the drift label).
     */
    default String lastSelectionTrigger() { return "initial"; }

    default String name() { return getClass().getSimpleName(); }
}