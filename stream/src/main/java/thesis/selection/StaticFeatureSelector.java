package thesis.selection;

import lombok.Getter;
import thesis.discretization.PiDDiscretizer;

import java.util.Arrays;
import java.util.Set;

public class StaticFeatureSelector implements FeatureSelector {

    @Getter private final int numFeatures;
    @Getter private final int numClasses;
    @Getter private final int k;
    private final PiDDiscretizer discretizer;
    private final RankerFactory rankerFactory;

    private int[] selection;
    private double[] initialScores;
    private String rankerName;
    @Getter private boolean initialized;
    @Getter private long ignoredNonFiniteRows;

    public StaticFeatureSelector(int numFeatures, int numClasses) {
        this(numFeatures, numClasses, defaultK(numFeatures),
                new PiDDiscretizer(numFeatures, numClasses),
                (f, b, c) -> new InformationGainRanker(f, b, c));
    }

    public StaticFeatureSelector(int numFeatures, int numClasses, int k,
                                 PiDDiscretizer discretizer,
                                 RankerFactory rankerFactory) {
        if (numFeatures < 1) throw new IllegalArgumentException("numFeatures must be >= 1");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        if (k < 1 || k > numFeatures) {
            throw new IllegalArgumentException("k must be in [1, " + numFeatures + "], got " + k);
        }
        if (discretizer == null) throw new IllegalArgumentException("discretizer must not be null");
        if (rankerFactory == null) throw new IllegalArgumentException("rankerFactory must not be null");
        if (discretizer.getNumFeatures() != numFeatures) {
            throw new IllegalArgumentException("discretizer numFeatures mismatch: " +
                    discretizer.getNumFeatures() + " vs " + numFeatures);
        }
        if (discretizer.getNumClasses() != numClasses) {
            throw new IllegalArgumentException("discretizer numClasses mismatch: " +
                    discretizer.getNumClasses() + " vs " + numClasses);
        }
        this.numFeatures = numFeatures;
        this.numClasses = numClasses;
        this.k = k;
        this.discretizer = discretizer;
        this.rankerFactory = rankerFactory;
        this.selection = null;
        this.initialScores = null;
        this.rankerName = null;
        this.initialized = false;
        this.ignoredNonFiniteRows = 0;
    }

    public static int defaultK(int numFeatures) {
        return Math.max(1, (int) Math.ceil(Math.sqrt(numFeatures)));
    }

    public void initializeIdentity() {
        if (initialized) {
            throw new IllegalStateException("selector already initialized; create a new instance to re-rank");
        }
        int m = Math.min(k, numFeatures);
        int[] sel = new int[m];
        for (int i = 0; i < m; i++) sel[i] = i;
        this.selection = sel;
        this.initialScores = new double[numFeatures];
        this.rankerName = "identity";
        this.initialized = true;
    }

    public void initializeWith(int[] explicitSelection) {
        if (initialized) {
            throw new IllegalStateException("selector already initialized; create a new instance to re-rank");
        }
        if (explicitSelection == null || explicitSelection.length != k) {
            throw new IllegalArgumentException("explicit selection must have length k=" + k);
        }
        boolean[] seen = new boolean[numFeatures];
        for (int idx : explicitSelection) {
            if (idx < 0 || idx >= numFeatures) {
                throw new IllegalArgumentException("selection index out of range: " + idx);
            }
            if (seen[idx]) throw new IllegalArgumentException("duplicate selection index: " + idx);
            seen[idx] = true;
        }
        int[] sorted = explicitSelection.clone();
        Arrays.sort(sorted);
        this.selection = sorted;
        this.initialScores = new double[numFeatures];
        this.rankerName = "explicit";
        this.initialized = true;
    }

    @Override
    public void initialize(double[][] initialWindow, int[] labels) {
        if (initialized) {
            throw new IllegalStateException("selector already initialized; create a new instance to re-rank");
        }
        if (initialWindow == null || labels == null) {
            throw new IllegalArgumentException("initialWindow and labels must not be null");
        }
        if (initialWindow.length != labels.length) {
            throw new IllegalArgumentException("initialWindow.length=" + initialWindow.length +
                    " != labels.length=" + labels.length);
        }
        if (initialWindow.length == 0) {
            throw new IllegalArgumentException("initialWindow must not be empty");
        }

        for (int i = 0; i < initialWindow.length; i++) {
            if (initialWindow[i] == null || initialWindow[i].length != numFeatures) {
                throw new IllegalArgumentException(
                        "row " + i + " has wrong feature count, expected " + numFeatures);
            }
            if (labels[i] < 0 || labels[i] >= numClasses) {
                throw new IllegalArgumentException(
                        "label[" + i + "]=" + labels[i] + " out of [0," + numClasses + ")");
            }
            if (rowAllFinite(initialWindow[i])) {
                discretizer.update(initialWindow[i], labels[i]);
            } else {
                ignoredNonFiniteRows++;
            }
        }

        if (!discretizer.isReady()) {
            throw new IllegalStateException(
                    "discretizer not ready after window of " + initialWindow.length +
                            " rows (ignored non-finite=" + ignoredNonFiniteRows +
                            ") — increase window size or lower warmupN");
        }

        FilterRanker ranker = rankerFactory.create(numFeatures, discretizer.getB2(), numClasses);
        if (ranker.getNumFeatures() != numFeatures) {
            throw new IllegalStateException(
                    "rankerFactory returned ranker with numFeatures=" + ranker.getNumFeatures() +
                            ", expected " + numFeatures);
        }

        for (int i = 0; i < initialWindow.length; i++) {
            if (!rowAllFinite(initialWindow[i])) continue;
            int[] bins = discretizer.discretizeAll(initialWindow[i]);
            ranker.update(bins, labels[i]);
        }

        this.initialScores = ranker.getFeatureScores();
        int[] top = ranker.selectTopK(k);
        Arrays.sort(top);
        this.selection = top;
        this.rankerName = ranker.name();
        this.initialized = true;
    }

    @Override
    public void update(double[] instance, int classLabel,
                       boolean driftAlarm, Set<Integer> driftingFeatures) {
        ensureInitialized();
        if (instance == null || instance.length != numFeatures) {
            throw new IllegalArgumentException(
                    "expected " + numFeatures + " features, got " +
                            (instance == null ? "null" : instance.length));
        }
        if (classLabel < 0 || classLabel >= numClasses) {
            throw new IllegalArgumentException("classLabel out of range: " + classLabel);
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
                    "expected " + numFeatures + " features, got " +
                            (fullInstance == null ? "null" : fullInstance.length));
        }
        double[] out = new double[selection.length];
        for (int i = 0; i < selection.length; i++) out[i] = fullInstance[selection[i]];
        return out;
    }

    @Override
    public double[] getInitialScores() {
        ensureInitialized();
        return initialScores == null ? null : Arrays.copyOf(initialScores, initialScores.length);
    }

    public String getRankerName() {
        ensureInitialized();
        return rankerName;
    }

    @Override
    public String name() { return "StaticFeatureSelector(K=" + k + ")"; }

    private void ensureInitialized() {
        if (!initialized) throw new IllegalStateException("selector not initialized — call initialize() first");
    }

    private static boolean rowAllFinite(double[] row) {
        for (double v : row) if (!Double.isFinite(v)) return false;
        return true;
    }
}