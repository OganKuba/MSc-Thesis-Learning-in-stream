package thesis.selection;

import thesis.discretization.PiDDiscretizer;

import java.util.Arrays;
import java.util.Set;
import java.util.function.BiFunction;

public class StaticFeatureSelector implements FeatureSelector {

    private final int numFeatures;
    private final int numClasses;
    private final int k;
    private final PiDDiscretizer discretizer;
    private final BiFunction<Integer, Integer, FilterRanker> rankerFactory;

    private int[] selection;
    private boolean initialized;

    public StaticFeatureSelector(int numFeatures, int numClasses) {
        this(numFeatures, numClasses, defaultK(numFeatures),
                new PiDDiscretizer(numFeatures, numClasses),
                (bins, classes) -> new InformationGainRanker(numFeatures, bins, classes));
    }

    public StaticFeatureSelector(int numFeatures, int numClasses, int k,
                                 PiDDiscretizer discretizer,
                                 BiFunction<Integer, Integer, FilterRanker> rankerFactory) {
        if (numFeatures < 1) throw new IllegalArgumentException("numFeatures must be >= 1");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        if (k < 1 || k > numFeatures) {
            throw new IllegalArgumentException("k must be in [1, " + numFeatures + "], got " + k);
        }
        if (discretizer.getNumFeatures() != numFeatures) {
            throw new IllegalArgumentException("discretizer numFeatures mismatch");
        }
        if (discretizer.getNumClasses() != numClasses) {
            throw new IllegalArgumentException("discretizer numClasses mismatch");
        }
        this.numFeatures = numFeatures;
        this.numClasses = numClasses;
        this.k = k;
        this.discretizer = discretizer;
        this.rankerFactory = rankerFactory;
        this.selection = null;
        this.initialized = false;
    }

    public static int defaultK(int numFeatures) {
        return Math.max(1, (int) Math.ceil(Math.sqrt(numFeatures)));
    }

    @Override
    public void initialize(double[][] initialWindow, int[] labels) {
        if (initialWindow == null || labels == null) {
            throw new IllegalArgumentException("initialWindow and labels must not be null");
        }
        if (initialWindow.length != labels.length) {
            throw new IllegalArgumentException("initialWindow and labels length mismatch");
        }
        if (initialWindow.length == 0) {
            throw new IllegalArgumentException("initialWindow must not be empty");
        }
        for (int i = 0; i < initialWindow.length; i++) {
            if (initialWindow[i].length != numFeatures) {
                throw new IllegalArgumentException(
                        "row " + i + " has " + initialWindow[i].length +
                                " features, expected " + numFeatures);
            }
            discretizer.update(initialWindow[i], labels[i]);
        }
        if (!discretizer.isReady()) {
            throw new IllegalStateException(
                    "discretizer not ready after initial window of " + initialWindow.length +
                            " — increase window size or lower warmupN");
        }

        FilterRanker ranker = rankerFactory.apply(discretizer.getB2(), numClasses);
        for (int i = 0; i < initialWindow.length; i++) {
            int[] bins = discretizer.discretizeAll(initialWindow[i]);
            ranker.update(bins, labels[i]);
        }

        int[] top = ranker.selectTopK(k);
        Arrays.sort(top);
        this.selection = top;
        this.initialized = true;
    }

    @Override
    public void update(double[] instance, int classLabel,
                       boolean driftAlarm, Set<Integer> driftingFeatures) {
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
        if (fullInstance.length != numFeatures) {
            throw new IllegalArgumentException(
                    "expected " + numFeatures + " features, got " + fullInstance.length);
        }
        double[] out = new double[selection.length];
        for (int i = 0; i < selection.length; i++) {
            out[i] = fullInstance[selection[i]];
        }
        return out;
    }

    @Override public int getNumFeatures()   { return numFeatures; }
    @Override public int getK()              { return k; }
    @Override public boolean isInitialized() { return initialized; }

    @Override
    public String name() {
        return "StaticFeatureSelector(K=" + k + ")";
    }

    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("selector not initialized — call initialize() first");
        }
    }
}