package thesis.selection;

import lombok.Getter;

import java.util.Arrays;
import java.util.Set;

/**
 * Baseline selector that preserves the complete input representation.
 *
 * <p>This is intentionally separate from {@link StaticFeatureSelector}: S1 ranks
 * features and keeps only the top-k subset, while NONE performs no feature
 * selection and lets the downstream learner use every original attribute.</p>
 */
public class NoFeatureSelection implements FeatureSelector {

    /** All features are always used, so the selection never changes. */
    @Override
    public String lastSelectionTrigger() { return "none"; }


    @Getter private final int numFeatures;
    @Getter private final int k;
    private final int[] selection;
    @Getter private boolean initialized;

    public NoFeatureSelection(int numFeatures) {
        if (numFeatures < 1) throw new IllegalArgumentException("numFeatures must be >= 1");
        this.numFeatures = numFeatures;
        this.k = numFeatures;
        this.selection = new int[numFeatures];
        for (int i = 0; i < numFeatures; i++) selection[i] = i;
        this.initialized = false;
    }

    @Override
    public void initialize(double[][] initialWindow, int[] labels) {
        if (initialized) {
            throw new IllegalStateException("selector already initialized");
        }
        if (initialWindow == null || labels == null) {
            throw new IllegalArgumentException("initialWindow and labels must not be null");
        }
        if (initialWindow.length != labels.length) {
            throw new IllegalArgumentException("initialWindow.length=" + initialWindow.length
                    + " != labels.length=" + labels.length);
        }
        for (int i = 0; i < initialWindow.length; i++) {
            if (initialWindow[i] == null || initialWindow[i].length != numFeatures) {
                throw new IllegalArgumentException(
                        "row " + i + " has wrong feature count, expected " + numFeatures);
            }
        }
        initialized = true;
    }

    @Override
    public void update(double[] instance, int classLabel,
                       boolean driftAlarm, Set<Integer> driftingFeatures) {
        ensureInitialized();
        if (instance == null || instance.length != numFeatures) {
            throw new IllegalArgumentException(
                    "expected " + numFeatures + " features, got "
                            + (instance == null ? "null" : instance.length));
        }
    }

    @Override
    public int[] getSelectedFeatures() {
        ensureInitialized();
        return Arrays.copyOf(selection, selection.length);
    }

    @Override
    public int[] getCurrentSelection() {
        return getSelectedFeatures();
    }

    @Override
    public double[] filterInstance(double[] fullInstance) {
        ensureInitialized();
        if (fullInstance == null || fullInstance.length != numFeatures) {
            throw new IllegalArgumentException(
                    "expected " + numFeatures + " features, got "
                            + (fullInstance == null ? "null" : fullInstance.length));
        }
        return Arrays.copyOf(fullInstance, fullInstance.length);
    }

    @Override
    public double[] getInitialScores() {
        ensureInitialized();
        double[] scores = new double[numFeatures];
        Arrays.fill(scores, 1.0);
        return scores;
    }

    @Override
    public String name() {
        return "NoFeatureSelection(K=" + k + ")";
    }

    private void ensureInitialized() {
        if (!initialized) throw new IllegalStateException("selector not initialized");
    }
}
