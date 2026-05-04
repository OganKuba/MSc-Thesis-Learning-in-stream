package thesis.selection;

import lombok.Getter;
import thesis.discretization.PiDDiscretizer;

import java.util.Arrays;
import java.util.Set;

public class AlarmTriggeredSelector implements FeatureSelector {

    public interface EventListener {
        void onAlarm(long alarmIndex, boolean accepted, Set<Integer> driftingFeatures);
        void onCollectingStart(int wPostDrift, int[] currentSelection);
        void onReSelection(int[] oldSelection, int[] newSelection,
                           double[] scores, boolean changed);
    }

    public static final double DEFAULT_DRIFT_DECAY = 0.05;
    public static final double DEFAULT_TIE_EPSILON = 0.01;

    @Getter private final int numFeatures;
    @Getter private final int numClasses;
    @Getter private final int k;
    @Getter private final int wPostDrift;
    @Getter private final double driftDecay;
    @Getter private final double tieEpsilon;
    private final PiDDiscretizer discretizer;
    private final RankerFactory rankerFactory;

    private FilterRanker ranker;
    private int[] selection;
    private double[] lastScores;
    @Getter private boolean initialized;
    @Getter private boolean collecting;
    @Getter private int collected;
    @Getter private long reSelections;
    @Getter private long alarmsObserved;
    @Getter private long alarmsAccepted;
    @Getter private long alarmsIgnoredWhileBusy;
    @Getter private long updatesBeforeInit;

    private EventListener listener;

    public AlarmTriggeredSelector(int numFeatures, int numClasses) {
        this(numFeatures, numClasses,
                StaticFeatureSelector.defaultK(numFeatures),
                500,
                new PiDDiscretizer(numFeatures, numClasses),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc),
                DEFAULT_DRIFT_DECAY,
                DEFAULT_TIE_EPSILON);
    }

    public AlarmTriggeredSelector(int numFeatures, int numClasses, int k, int wPostDrift,
                                  PiDDiscretizer discretizer,
                                  RankerFactory rankerFactory) {
        this(numFeatures, numClasses, k, wPostDrift, discretizer, rankerFactory,
                DEFAULT_DRIFT_DECAY, DEFAULT_TIE_EPSILON);
    }

    public AlarmTriggeredSelector(int numFeatures, int numClasses, int k, int wPostDrift,
                                  PiDDiscretizer discretizer,
                                  RankerFactory rankerFactory,
                                  double driftDecay) {
        this(numFeatures, numClasses, k, wPostDrift, discretizer, rankerFactory,
                driftDecay, DEFAULT_TIE_EPSILON);
    }

    public AlarmTriggeredSelector(int numFeatures, int numClasses, int k, int wPostDrift,
                                  PiDDiscretizer discretizer,
                                  RankerFactory rankerFactory,
                                  double driftDecay,
                                  double tieEpsilon) {
        if (numFeatures < 1) throw new IllegalArgumentException("numFeatures must be >= 1");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        if (k < 1 || k > numFeatures)
            throw new IllegalArgumentException("k must be in [1, " + numFeatures + "], got " + k);
        if (wPostDrift < 50)
            throw new IllegalArgumentException("wPostDrift must be >= 50, got " + wPostDrift);
        if (discretizer == null) throw new IllegalArgumentException("discretizer must not be null");
        if (rankerFactory == null) throw new IllegalArgumentException("rankerFactory must not be null");
        if (discretizer.getNumFeatures() != numFeatures)
            throw new IllegalArgumentException("discretizer numFeatures mismatch");
        if (discretizer.getNumClasses() != numClasses)
            throw new IllegalArgumentException("discretizer numClasses mismatch");
        if (!(driftDecay > 0.0 && driftDecay <= 1.0))
            throw new IllegalArgumentException("driftDecay must be in (0,1], got " + driftDecay);
        if (!(tieEpsilon >= 0.0))
            throw new IllegalArgumentException("tieEpsilon must be >= 0, got " + tieEpsilon);
        this.numFeatures = numFeatures;
        this.numClasses = numClasses;
        this.k = k;
        this.wPostDrift = wPostDrift;
        this.discretizer = discretizer;
        this.rankerFactory = rankerFactory;
        this.driftDecay = driftDecay;
        this.tieEpsilon = tieEpsilon;
    }

    public void setEventListener(EventListener listener) {
        this.listener = listener;
    }

    @Override
    public void initialize(double[][] initialWindow, int[] labels) {
        if (initialized) {
            throw new IllegalStateException("selector already initialized");
        }
        if (initialWindow == null || labels == null
                || initialWindow.length != labels.length || initialWindow.length == 0)
            throw new IllegalArgumentException("invalid initialWindow / labels");

        for (int i = 0; i < initialWindow.length; i++) {
            if (initialWindow[i] == null || initialWindow[i].length != numFeatures)
                throw new IllegalArgumentException("row " + i + " has wrong feature count");
            if (labels[i] < 0 || labels[i] >= numClasses)
                throw new IllegalArgumentException(
                        "label[" + i + "]=" + labels[i] + " out of [0," + numClasses + ")");
            if (allFinite(initialWindow[i])) discretizer.update(initialWindow[i], labels[i]);
        }
        if (!discretizer.isReady())
            throw new IllegalStateException("discretizer not ready after initial window");

        ranker = rankerFactory.create(numFeatures, discretizer.getB2(), numClasses);
        if (ranker.getNumFeatures() != numFeatures) {
            throw new IllegalStateException("rankerFactory returned ranker with wrong numFeatures: "
                    + ranker.getNumFeatures() + " vs " + numFeatures);
        }
        for (int i = 0; i < initialWindow.length; i++) {
            if (!allFinite(initialWindow[i])) continue;
            ranker.update(discretizer.discretizeAll(initialWindow[i]), labels[i]);
        }

        this.lastScores = ranker.getFeatureScores();
        int[] top = ranker.selectTopK(k);
        Arrays.sort(top);
        this.selection = top;
        this.initialized = true;
        this.collecting = false;
        this.collected = 0;
        this.reSelections = 0;
        this.alarmsObserved = 0;
        this.alarmsAccepted = 0;
        this.alarmsIgnoredWhileBusy = 0;
        this.updatesBeforeInit = 0;
    }

    @Override
    public void update(double[] instance, int classLabel,
                       boolean driftAlarm, Set<Integer> driftingFeatures) {
        if (!initialized) {
            updatesBeforeInit++;
            return;
        }
        if (instance == null || instance.length != numFeatures)
            throw new IllegalArgumentException("expected " + numFeatures + " features");
        if (classLabel < 0 || classLabel >= numClasses)
            throw new IllegalArgumentException(
                    "classLabel=" + classLabel + " out of [0," + numClasses + ")");

        if (driftAlarm) {
            alarmsObserved++;
            if (collecting) {
                alarmsIgnoredWhileBusy++;
                if (listener != null) listener.onAlarm(alarmsObserved, false, driftingFeatures);
            } else {
                alarmsAccepted++;
                applyDriftDecay(driftingFeatures);
                ranker.reset();
                collecting = true;
                collected = 0;
                if (listener != null) {
                    listener.onAlarm(alarmsObserved, true, driftingFeatures);
                    listener.onCollectingStart(wPostDrift, selection.clone());
                }
            }
        }

        if (allFinite(instance)) {
            discretizer.update(instance, classLabel);
        }

        if (collecting) {
            if (allFinite(instance) && discretizer.isReady()) {
                ranker.update(discretizer.discretizeAll(instance), classLabel);
            }
            collected++;
            if (collected >= wPostDrift) {
                int[] oldSel = selection;
                lastScores = ranker.getFeatureScores();
                int[] top = ranker.selectTopK(k, oldSel, tieEpsilon);
                Arrays.sort(top);
                selection = top;
                collecting = false;
                collected = 0;
                reSelections++;
                boolean changed = !Arrays.equals(oldSel, selection);
                if (listener != null) {
                    listener.onReSelection(oldSel, selection.clone(),
                            lastScores.clone(), changed);
                }
            }
        }
    }

    private void applyDriftDecay(Set<Integer> driftingFeatures) {
        if (driftingFeatures != null && !driftingFeatures.isEmpty()) {
            for (int idx : driftingFeatures) {
                if (idx >= 0 && idx < numFeatures) {
                    discretizer.softResetFeature(idx, driftDecay);
                }
            }
        } else {
            for (int i = 0; i < numFeatures; i++) {
                if (discretizer.isReady(i)) discretizer.softResetFeature(i, driftDecay);
            }
        }
    }

    @Override
    public int[] getSelectedFeatures() {
        ensureInitialized();
        return Arrays.copyOf(selection, selection.length);
    }

    @Override
    public int[] getCurrentSelection() { return getSelectedFeatures(); }

    @Override
    public double[] filterInstance(double[] fullInstance) {
        ensureInitialized();
        if (fullInstance == null || fullInstance.length != numFeatures)
            throw new IllegalArgumentException("expected " + numFeatures + " features");
        double[] out = new double[selection.length];
        for (int i = 0; i < selection.length; i++) out[i] = fullInstance[selection[i]];
        return out;
    }

    public double[] getLastScores() {
        return lastScores == null ? null : Arrays.copyOf(lastScores, lastScores.length);
    }

    public boolean isCollectingPostDrift() { return collecting; }
    public int getPostDriftCount() { return collected; }

    @Override
    public String name() {
        return "AlarmTriggeredSelector(K=" + k + ", W=" + wPostDrift +
                ", decay=" + driftDecay + ", tieEps=" + tieEpsilon + ")";
    }

    private void ensureInitialized() {
        if (!initialized) throw new IllegalStateException("selector not initialized");
    }

    private static boolean allFinite(double[] row) {
        for (double v : row) if (!Double.isFinite(v)) return false;
        return true;
    }
}