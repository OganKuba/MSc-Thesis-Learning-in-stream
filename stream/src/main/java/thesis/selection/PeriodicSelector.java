package thesis.selection;

import lombok.Getter;
import thesis.discretization.PiDDiscretizer;

import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;

public class PeriodicSelector implements FeatureSelector {

    public interface EventListener {
        void onPeriodicTick(long instanceNum, boolean triggered);
        void onReSelection(long instanceNum,
                           int[] oldSelection, int[] newSelection,
                           Set<Integer> replacedOut, Set<Integer> replacedIn,
                           double[] scores, long[] tenure,
                           double stabilityRatio);
    }

    public static final double DEFAULT_TIE_EPSILON = 0.01;
    public static final double DEFAULT_MAX_SWAP_FRACTION = 0.3;

    @Getter private final int numFeatures;
    @Getter private final int numClasses;
    @Getter private final int k;
    @Getter private final int periodN;
    @Getter private final int minTenure;
    @Getter private final int maxSwapsPerCycle;
    @Getter private final double tieEpsilon;
    private final PiDDiscretizer discretizer;
    private final RankerFactory rankerFactory;

    private int[] selection;
    private long[] tenure;
    @Getter private long updates;

    private final int[][] ringBins;
    private final int[] ringLabels;
    private int ringPos;
    private int ringCount;

    @Getter private boolean initialized;
    @Getter private long swapEvents;
    @Getter private long swappedFeatures;
    @Getter private long reSelectionTicks;
    @Getter private long updatesBeforeInit;

    private EventListener listener;

    public PeriodicSelector(int numFeatures, int numClasses) {
        this(numFeatures, numClasses,
                StaticFeatureSelector.defaultK(numFeatures),
                1000, 500,
                new PiDDiscretizer(numFeatures, numClasses),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc),
                DEFAULT_TIE_EPSILON,
                DEFAULT_MAX_SWAP_FRACTION);
    }

    public PeriodicSelector(int numFeatures, int numClasses, int k,
                            int periodN, int minTenure,
                            PiDDiscretizer discretizer,
                            RankerFactory rankerFactory) {
        this(numFeatures, numClasses, k, periodN, minTenure, discretizer, rankerFactory,
                DEFAULT_TIE_EPSILON, DEFAULT_MAX_SWAP_FRACTION);
    }

    public PeriodicSelector(int numFeatures, int numClasses, int k,
                            int periodN, int minTenure,
                            PiDDiscretizer discretizer,
                            RankerFactory rankerFactory,
                            double tieEpsilon,
                            double maxSwapFraction) {
        if (numFeatures < 1) throw new IllegalArgumentException("numFeatures must be >= 1");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        if (k < 1 || k > numFeatures)
            throw new IllegalArgumentException("k must be in [1, " + numFeatures + "], got " + k);
        if (periodN < 100) throw new IllegalArgumentException("periodN must be >= 100, got " + periodN);
        if (minTenure < 0) throw new IllegalArgumentException("minTenure must be >= 0");
        if (discretizer == null) throw new IllegalArgumentException("discretizer must not be null");
        if (rankerFactory == null) throw new IllegalArgumentException("rankerFactory must not be null");
        if (discretizer.getNumFeatures() != numFeatures)
            throw new IllegalArgumentException("discretizer numFeatures mismatch");
        if (discretizer.getNumClasses() != numClasses)
            throw new IllegalArgumentException("discretizer numClasses mismatch");
        if (!(tieEpsilon >= 0.0))
            throw new IllegalArgumentException("tieEpsilon must be >= 0");
        if (!(maxSwapFraction > 0.0 && maxSwapFraction <= 1.0))
            throw new IllegalArgumentException("maxSwapFraction must be in (0,1]");
        this.numFeatures = numFeatures;
        this.numClasses = numClasses;
        this.k = k;
        this.periodN = periodN;
        this.minTenure = minTenure;
        this.maxSwapsPerCycle = Math.max(1, (int) Math.ceil(k * maxSwapFraction));
        this.tieEpsilon = tieEpsilon;
        this.discretizer = discretizer;
        this.rankerFactory = rankerFactory;
        this.ringBins = new int[periodN][];
        this.ringLabels = new int[periodN];
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
                || initialWindow.length != labels.length || initialWindow.length == 0) {
            throw new IllegalArgumentException("invalid initialWindow / labels");
        }
        for (int i = 0; i < initialWindow.length; i++) {
            if (initialWindow[i] == null || initialWindow[i].length != numFeatures)
                throw new IllegalArgumentException("row " + i + " has wrong feature count");
            if (labels[i] < 0 || labels[i] >= numClasses)
                throw new IllegalArgumentException(
                        "label[" + i + "]=" + labels[i] + " out of [0," + numClasses + ")");
            if (allFinite(initialWindow[i])) discretizer.update(initialWindow[i], labels[i]);
        }
        if (!discretizer.isReady()) {
            throw new IllegalStateException("discretizer not ready after initial window");
        }

        FilterRanker ranker = rankerFactory.create(numFeatures, discretizer.getB2(), numClasses);
        if (ranker.getNumFeatures() != numFeatures) {
            throw new IllegalStateException("rankerFactory returned ranker with wrong numFeatures");
        }
        for (int i = 0; i < initialWindow.length; i++) {
            if (!allFinite(initialWindow[i])) continue;
            int[] bins = discretizer.discretizeAll(initialWindow[i]);
            ranker.update(bins, labels[i]);
            pushRing(bins, labels[i]);
        }

        int[] top = ranker.selectTopK(k);
        Arrays.sort(top);
        this.selection = top;
        this.tenure = new long[numFeatures];
        for (int idx : top) tenure[idx] = Long.MAX_VALUE / 2;
        this.updates = 0;
        this.initialized = true;
        this.swapEvents = 0;
        this.swappedFeatures = 0;
        this.reSelectionTicks = 0;
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

        if (allFinite(instance)) discretizer.update(instance, classLabel);
        if (!discretizer.isReady()) return;

        int[] bins = discretizer.discretizeAll(instance);
        pushRing(bins, classLabel);
        updates++;
        for (int idx : selection) {
            if (tenure[idx] < Long.MAX_VALUE / 2) tenure[idx]++;
        }

        boolean trigger = updates % periodN == 0;
        if (listener != null) listener.onPeriodicTick(updates, trigger);
        if (trigger) reSelect();
    }

    private void reSelect() {
        reSelectionTicks++;
        FilterRanker ranker = rankerFactory.create(numFeatures, discretizer.getB2(), numClasses);
        for (int i = 0; i < ringCount; i++) {
            ranker.update(ringBins[i], ringLabels[i]);
        }
        double[] scores = ranker.getFeatureScores();

        boolean[] inSel = new boolean[numFeatures];
        for (int idx : selection) inSel[idx] = true;

        int[] outCandidates = filteredSortedAsc(selection, scores, true);
        int[] inCandidates = sortedDesc(notInSet(inSel), scores);

        Set<Integer> newSet = new LinkedHashSet<>();
        for (int idx : selection) newSet.add(idx);

        Set<Integer> replacedOut = new LinkedHashSet<>();
        Set<Integer> replacedIn = new LinkedHashSet<>();

        int swaps = 0;
        int maxS = Math.min(maxSwapsPerCycle, Math.min(outCandidates.length, inCandidates.length));
        for (int s = 0; s < maxS; s++) {
            int outIdx = outCandidates[s];
            int inIdx = inCandidates[s];
            if (scores[inIdx] > scores[outIdx] + tieEpsilon) {
                newSet.remove(outIdx);
                newSet.add(inIdx);
                tenure[inIdx] = 0;
                replacedOut.add(outIdx);
                replacedIn.add(inIdx);
                swaps++;
            } else {
                break;
            }
        }

        int[] oldSelection = selection;
        if (swaps > 0) {
            int[] newSelection = new int[k];
            int idx = 0;
            for (int f : newSet) newSelection[idx++] = f;
            Arrays.sort(newSelection);
            this.selection = newSelection;
            this.swapEvents++;
            this.swappedFeatures += swaps;
        }
        if (listener != null) {
            double stability = stabilityRatio(oldSelection, selection);
            listener.onReSelection(updates, oldSelection, selection.clone(),
                    replacedOut, replacedIn, scores.clone(), tenure.clone(), stability);
        }
    }

    private double stabilityRatio(int[] oldSel, int[] newSel) {
        Set<Integer> oldSet = new HashSet<>();
        for (int i : oldSel) oldSet.add(i);
        int kept = 0;
        for (int i : newSel) if (oldSet.contains(i)) kept++;
        return (double) kept / oldSel.length;
    }

    private int[] filteredSortedAsc(int[] candidates, double[] scores, boolean filterTenure) {
        int n = candidates.length;
        int[] tmp = new int[n];
        int j = 0;
        for (int idx : candidates) {
            if (!filterTenure || tenure[idx] >= minTenure) tmp[j++] = idx;
        }
        int[] out = Arrays.copyOf(tmp, j);
        sortStable(out, scores, true);
        return out;
    }

    private int[] sortedDesc(int[] candidates, double[] scores) {
        int[] out = candidates.clone();
        sortStable(out, scores, false);
        return out;
    }

    private static void sortStable(int[] arr, double[] scores, boolean ascending) {
        Integer[] boxed = new Integer[arr.length];
        for (int i = 0; i < arr.length; i++) boxed[i] = arr[i];
        Arrays.sort(boxed, (a, b) -> {
            int cmp = Double.compare(scores[a], scores[b]);
            if (!ascending) cmp = -cmp;
            if (cmp != 0) return cmp;
            return Integer.compare(a, b);
        });
        for (int i = 0; i < arr.length; i++) arr[i] = boxed[i];
    }

    private int[] notInSet(boolean[] inSel) {
        int n = 0;
        for (boolean b : inSel) if (!b) n++;
        int[] out = new int[n];
        int j = 0;
        for (int f = 0; f < numFeatures; f++) if (!inSel[f]) out[j++] = f;
        return out;
    }

    private void pushRing(int[] bins, int label) {
        ringBins[ringPos] = bins;
        ringLabels[ringPos] = label;
        ringPos = (ringPos + 1) % periodN;
        if (ringCount < periodN) ringCount++;
    }

    public int getRingSize() { return ringCount; }

    public long[] getTenureSnapshot() {
        ensureInitialized();
        return tenure.clone();
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

    @Override
    public String name() {
        return "PeriodicSelector(K=" + k + ", N=" + periodN + ", N_min=" + minTenure +
                ", maxSwap=" + maxSwapsPerCycle + ", tieEps=" + tieEpsilon + ")";
    }

    private void ensureInitialized() {
        if (!initialized) throw new IllegalStateException("selector not initialized");
    }

    private static boolean allFinite(double[] row) {
        for (double v : row) if (!Double.isFinite(v)) return false;
        return true;
    }
}