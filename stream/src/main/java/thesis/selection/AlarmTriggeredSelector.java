package thesis.selection;

import lombok.Getter;
import thesis.discretization.PiDDiscretizer;

import java.util.Arrays;
import java.util.Set;
import java.util.function.BiFunction;

@Getter
public class AlarmTriggeredSelector implements FeatureSelector {

    private final int numFeatures;
    private final int numClasses;
    private final int k;
    private final int wPostDrift;
    private final PiDDiscretizer discretizer;
    private final BiFunction<Integer, Integer, FilterRanker> rankerFactory;

    private FilterRanker ranker;
    private int[] selection;
    private boolean initialized;
    private boolean collecting;
    private int collected;
    private long reSelections;

    public AlarmTriggeredSelector(int numFeatures, int numClasses) {
        this(numFeatures, numClasses,
                StaticFeatureSelector.defaultK(numFeatures),
                500,
                new PiDDiscretizer(numFeatures, numClasses),
                (bins, classes) -> new InformationGainRanker(numFeatures, bins, classes));
    }

    public AlarmTriggeredSelector(int numFeatures, int numClasses, int k, int wPostDrift,
                                  PiDDiscretizer discretizer,
                                  BiFunction<Integer, Integer, FilterRanker> rankerFactory) {
        if (numFeatures < 1) throw new IllegalArgumentException("numFeatures must be >= 1");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        if (k < 1 || k > numFeatures) {
            throw new IllegalArgumentException("k must be in [1, " + numFeatures + "]");
        }
        if (wPostDrift < 50) throw new IllegalArgumentException("wPostDrift must be >= 50");
        if (discretizer.getNumFeatures() != numFeatures) {
            throw new IllegalArgumentException("discretizer numFeatures mismatch");
        }
        this.numFeatures = numFeatures;
        this.numClasses = numClasses;
        this.k = k;
        this.wPostDrift = wPostDrift;
        this.discretizer = discretizer;
        this.rankerFactory = rankerFactory;
    }

    @Override
    public void initialize(double[][] initialWindow, int[] labels) {
        if (initialWindow == null || labels == null
                || initialWindow.length != labels.length || initialWindow.length == 0) {
            throw new IllegalArgumentException("invalid initialWindow / labels");
        }
        for (int i = 0; i < initialWindow.length; i++) {
            if (initialWindow[i].length != numFeatures) {
                throw new IllegalArgumentException("row " + i + " has wrong feature count");
            }
            discretizer.update(initialWindow[i], labels[i]);
        }
        if (!discretizer.isReady()) {
            throw new IllegalStateException("discretizer not ready after initial window");
        }

        ranker = rankerFactory.apply(discretizer.getB2(), numClasses);
        for (int i = 0; i < initialWindow.length; i++) {
            ranker.update(discretizer.discretizeAll(initialWindow[i]), labels[i]);
        }

        int[] top = ranker.selectTopK(k);
        Arrays.sort(top);
        this.selection = top;
        this.initialized = true;
        this.collecting = false;
        this.collected = 0;
        this.reSelections = 0;
    }

    @Override
    public void update(double[] instance, int classLabel,
                       boolean driftAlarm, Set<Integer> driftingFeatures) {
        if (!initialized) return;
        if (instance.length != numFeatures) {
            throw new IllegalArgumentException("expected " + numFeatures + " features");
        }

        discretizer.update(instance, classLabel);

        if (driftAlarm && !collecting) {
            ranker.reset();
            collecting = true;
            collected = 0;
        }

        if (collecting && discretizer.isReady()) {
            ranker.update(discretizer.discretizeAll(instance), classLabel);
            collected++;
            if (collected >= wPostDrift) {
                int[] top = ranker.selectTopK(k);
                Arrays.sort(top);
                this.selection = top;
                this.collecting = false;
                this.collected = 0;
                this.reSelections++;
            }
        }
    }

    @Override
    public int[] getSelectedFeatures()   { ensureInitialized(); return Arrays.copyOf(selection, selection.length); }
    @Override
    public int[] getCurrentSelection()   { return getSelectedFeatures(); }

    @Override
    public double[] filterInstance(double[] fullInstance) {
        ensureInitialized();
        if (fullInstance.length != numFeatures) {
            throw new IllegalArgumentException("expected " + numFeatures + " features");
        }
        double[] out = new double[selection.length];
        for (int i = 0; i < selection.length; i++) out[i] = fullInstance[selection[i]];
        return out;
    }

    @Override public int getNumFeatures()    { return numFeatures; }
    @Override public int getK()              { return k; }
    @Override public boolean isInitialized() { return initialized; }
    @Override public String name() {
        return "AlarmTriggeredSelector(K=" + k + ", W=" + wPostDrift + ")";
    }

    private void ensureInitialized() {
        if (!initialized) throw new IllegalStateException("selector not initialized");
    }
}