package thesis.selection;

import lombok.Getter;
import thesis.discretization.PiDDiscretizer;

import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;

public class DriftAwareSelector implements FeatureSelector {

    public enum TriggerType { NONE, PERIODIC, ALARM_WHERE }

    public interface EventListener {
        void onAlarm(long alarmIdx, boolean accepted, Set<Integer> driftingFeatures,
                     Set<Integer> driftingSelected, Set<Integer> stableSelected);
        void onPeriodicTick(long instanceNum, boolean triggered);
        void onSwap(TriggerType trigger, long instanceNum,
                    int[] oldSelection, int[] newSelection,
                    Set<Integer> replacedOut, Set<Integer> replacedIn,
                    double[] scores, long[] tenure, double stabilityRatio);
    }

    public static final double DEFAULT_TIE_EPSILON = 0.01;
    public static final double DEFAULT_DRIFT_DECAY = 0.05;
    public static final double DEFAULT_MAX_SWAP_FRACTION = 0.3;

    @Getter private final int numFeatures;
    @Getter private final int numClasses;
    @Getter private final int k;
    @Getter private final int periodN;
    @Getter private final int minTenure;
    @Getter private final int wPostDrift;
    @Getter private final int maxSwapsPerCycle;
    @Getter private final double tieEpsilon;
    @Getter private final double driftDecay;
    private final PiDDiscretizer discretizer;
    private final RankerFactory rankerFactory;

    private int[] selection;
    private long[] tenure;
    @Getter private long updates;

    private final int[][] ringBins;
    private final int[] ringLabels;
    private int ringPos;
    private int ringCount;

    @Getter private boolean collecting;
    @Getter private int collected;
    private FilterRanker postDriftRanker;
    private Set<Integer> pendingDriftingFeatures;

    @Getter private boolean initialized;
    @Getter private long periodicSwapEvents;
    @Getter private long alarmSwapEvents;
    @Getter private long swappedByAlarm;
    @Getter private long swappedByPeriodic;
    @Getter private long alarmsObserved;
    @Getter private long alarmsAccepted;
    @Getter private long alarmsIgnoredWhileBusy;
    @Getter private long alarmsIgnoredNoTargets;
    @Getter private long alarmsIgnoredAllStable;
    @Getter private long updatesBeforeInit;

    private EventListener listener;

    public DriftAwareSelector(int numFeatures, int numClasses) {
        this(numFeatures, numClasses,
                StaticFeatureSelector.defaultK(numFeatures),
                1000, 500, 500,
                new PiDDiscretizer(numFeatures, numClasses),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc),
                DEFAULT_TIE_EPSILON, DEFAULT_DRIFT_DECAY, DEFAULT_MAX_SWAP_FRACTION);
    }

    public DriftAwareSelector(int numFeatures, int numClasses, int k,
                              int periodN, int minTenure, int wPostDrift,
                              PiDDiscretizer discretizer,
                              RankerFactory rankerFactory) {
        this(numFeatures, numClasses, k, periodN, minTenure, wPostDrift,
                discretizer, rankerFactory,
                DEFAULT_TIE_EPSILON, DEFAULT_DRIFT_DECAY, DEFAULT_MAX_SWAP_FRACTION);
    }

    public DriftAwareSelector(int numFeatures, int numClasses, int k,
                              int periodN, int minTenure, int wPostDrift,
                              PiDDiscretizer discretizer,
                              RankerFactory rankerFactory,
                              double tieEpsilon, double driftDecay, double maxSwapFraction) {
        if (numFeatures < 1) throw new IllegalArgumentException("numFeatures must be >= 1");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        if (k < 1 || k > numFeatures)
            throw new IllegalArgumentException("k must be in [1, " + numFeatures + "], got " + k);
        if (periodN < 100) throw new IllegalArgumentException("periodN must be >= 100");
        if (minTenure < 0) throw new IllegalArgumentException("minTenure must be >= 0");
        if (wPostDrift < 50) throw new IllegalArgumentException("wPostDrift must be >= 50");
        if (discretizer == null) throw new IllegalArgumentException("discretizer must not be null");
        if (rankerFactory == null) throw new IllegalArgumentException("rankerFactory must not be null");
        if (discretizer.getNumFeatures() != numFeatures)
            throw new IllegalArgumentException("discretizer numFeatures mismatch");
        if (discretizer.getNumClasses() != numClasses)
            throw new IllegalArgumentException("discretizer numClasses mismatch");
        if (!(tieEpsilon >= 0.0))
            throw new IllegalArgumentException("tieEpsilon must be >= 0");
        if (!(driftDecay > 0.0 && driftDecay <= 1.0))
            throw new IllegalArgumentException("driftDecay must be in (0,1]");
        if (!(maxSwapFraction > 0.0 && maxSwapFraction <= 1.0))
            throw new IllegalArgumentException("maxSwapFraction must be in (0,1]");
        this.numFeatures = numFeatures;
        this.numClasses = numClasses;
        this.k = k;
        this.periodN = periodN;
        this.minTenure = minTenure;
        this.wPostDrift = wPostDrift;
        this.maxSwapsPerCycle = Math.max(1, (int) Math.ceil(k * maxSwapFraction));
        this.tieEpsilon = tieEpsilon;
        this.driftDecay = driftDecay;
        this.discretizer = discretizer;
        this.rankerFactory = rankerFactory;
        this.ringBins = new int[periodN][];
        this.ringLabels = new int[periodN];
    }

    public void setEventListener(EventListener listener) { this.listener = listener; }

    @Override
    public void initialize(double[][] initialWindow, int[] labels) {
        if (initialized) throw new IllegalStateException("selector already initialized");
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
        if (!discretizer.isReady())
            throw new IllegalStateException("discretizer not ready after initial window");

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
        this.collecting = false;
        this.collected = 0;
        this.postDriftRanker = null;
        this.pendingDriftingFeatures = null;
        this.periodicSwapEvents = 0;
        this.alarmSwapEvents = 0;
        this.swappedByAlarm = 0;
        this.swappedByPeriodic = 0;
        this.alarmsObserved = 0;
        this.alarmsAccepted = 0;
        this.alarmsIgnoredWhileBusy = 0;
        this.alarmsIgnoredNoTargets = 0;
        this.alarmsIgnoredAllStable = 0;
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

        if (driftAlarm) handleAlarm(driftingFeatures);

        if (allFinite(instance)) discretizer.update(instance, classLabel);
        if (!discretizer.isReady()) return;

        int[] bins = discretizer.discretizeAll(instance);
        pushRing(bins, classLabel);
        updates++;
        for (int idx : selection) {
            if (tenure[idx] < Long.MAX_VALUE / 2) tenure[idx]++;
        }

        if (collecting) {
            postDriftRanker.update(bins, classLabel);
            collected++;
            if (collected >= wPostDrift) {
                alarmSwap();
                collecting = false;
                collected = 0;
                postDriftRanker = null;
                pendingDriftingFeatures = null;
            }
        }

        boolean trigger = updates % periodN == 0;
        if (listener != null) listener.onPeriodicTick(updates, trigger);
        if (trigger) periodicReSelect();
    }

    private void handleAlarm(Set<Integer> driftingFeatures) {
        alarmsObserved++;
        if (collecting) {
            alarmsIgnoredWhileBusy++;
            emitAlarm(false, driftingFeatures, Collections.<Integer>emptySet(),
                    Collections.<Integer>emptySet());
            return;
        }
        Set<Integer> sanitized = sanitizeDrifting(driftingFeatures);
        if (sanitized.isEmpty()) {
            alarmsIgnoredNoTargets++;
            emitAlarm(false, driftingFeatures, Collections.<Integer>emptySet(),
                    selectionAsSet());
            return;
        }
        Set<Integer> selSet = selectionAsSet();
        Set<Integer> driftingSelected = new LinkedHashSet<>();
        for (int idx : sanitized) if (selSet.contains(idx)) driftingSelected.add(idx);
        Set<Integer> stableSelected = new LinkedHashSet<>();
        for (int idx : selection) if (!sanitized.contains(idx)) stableSelected.add(idx);
        if (driftingSelected.isEmpty()) {
            alarmsIgnoredAllStable++;
            emitAlarm(false, driftingFeatures, driftingSelected, stableSelected);
            return;
        }
        for (int idx : sanitized) discretizer.softResetFeature(idx, driftDecay);
        postDriftRanker = rankerFactory.create(numFeatures, discretizer.getB2(), numClasses);
        collecting = true;
        collected = 0;
        pendingDriftingFeatures = sanitized;
        alarmsAccepted++;
        emitAlarm(true, driftingFeatures, driftingSelected, stableSelected);
    }

    private Set<Integer> sanitizeDrifting(Set<Integer> driftingFeatures) {
        Set<Integer> out = new LinkedHashSet<>();
        if (driftingFeatures == null) return out;
        for (Integer idx : driftingFeatures) {
            if (idx != null && idx >= 0 && idx < numFeatures) out.add(idx);
        }
        return out;
    }

    private Set<Integer> selectionAsSet() {
        Set<Integer> s = new LinkedHashSet<>();
        for (int i : selection) s.add(i);
        return s;
    }

    private void emitAlarm(boolean accepted, Set<Integer> raw,
                           Set<Integer> driftingSelected, Set<Integer> stableSelected) {
        if (listener != null) {
            listener.onAlarm(alarmsObserved, accepted,
                    raw == null ? Collections.<Integer>emptySet() : raw,
                    driftingSelected, stableSelected);
        }
    }

    private void alarmSwap() {
        Set<Integer> targets = pendingDriftingFeatures;
        if (targets == null || targets.isEmpty()) return;

        Set<Integer> selSet = selectionAsSet();
        int[] outCandidates = filterEligibleOut(selection, targets);
        if (outCandidates.length == 0) return;

        int[] inCandidates = filterEligibleIn(selSet, targets);
        if (inCandidates.length == 0) return;

        double[] scores = postDriftRanker.getFeatureScores();
        sortByScoreStable(outCandidates, scores, true);
        sortByScoreStable(inCandidates, scores, false);

        Set<Integer> newSet = new LinkedHashSet<>();
        for (int idx : selection) newSet.add(idx);
        Set<Integer> replacedOut = new LinkedHashSet<>();
        Set<Integer> replacedIn = new LinkedHashSet<>();

        int swaps = 0;
        int maxS = Math.min(maxSwapsPerCycle,
                Math.min(targets.size(), Math.min(outCandidates.length, inCandidates.length)));
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

        int[] oldSel = selection;
        if (swaps > 0) {
            commitSelection(newSet);
            alarmSwapEvents++;
            swappedByAlarm += swaps;
        }
        if (listener != null) {
            listener.onSwap(TriggerType.ALARM_WHERE, updates, oldSel, selection.clone(),
                    replacedOut, replacedIn, scores.clone(), tenure.clone(),
                    stabilityRatio(oldSel, selection));
        }
    }

    private int[] filterEligibleOut(int[] sel, Set<Integer> targets) {
        int n = 0;
        for (int idx : sel) if (targets.contains(idx) && tenure[idx] >= minTenure) n++;
        int[] out = new int[n];
        int j = 0;
        for (int idx : sel) if (targets.contains(idx) && tenure[idx] >= minTenure) out[j++] = idx;
        return out;
    }

    private int[] filterEligibleIn(Set<Integer> selSet, Set<Integer> blocked) {
        int n = 0;
        for (int f = 0; f < numFeatures; f++) {
            if (!selSet.contains(f) && !blocked.contains(f)) n++;
        }
        int[] out = new int[n];
        int j = 0;
        for (int f = 0; f < numFeatures; f++) {
            if (!selSet.contains(f) && !blocked.contains(f)) out[j++] = f;
        }
        return out;
    }

    private void periodicReSelect() {
        FilterRanker ranker = rankerFactory.create(numFeatures, discretizer.getB2(), numClasses);
        for (int i = 0; i < ringCount; i++) ranker.update(ringBins[i], ringLabels[i]);
        double[] scores = ranker.getFeatureScores();

        Set<Integer> selSet = selectionAsSet();
        int outCnt = 0;
        for (int idx : selection) if (tenure[idx] >= minTenure) outCnt++;
        if (outCnt == 0) {
            emitPeriodicNoSwap(scores);
            return;
        }
        int[] outCandidates = new int[outCnt];
        int j = 0;
        for (int idx : selection) if (tenure[idx] >= minTenure) outCandidates[j++] = idx;

        int inCnt = numFeatures - selection.length;
        int[] inCandidates = new int[inCnt];
        int j2 = 0;
        for (int f = 0; f < numFeatures; f++) if (!selSet.contains(f)) inCandidates[j2++] = f;

        sortByScoreStable(outCandidates, scores, true);
        sortByScoreStable(inCandidates, scores, false);

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

        int[] oldSel = selection;
        if (swaps > 0) {
            commitSelection(newSet);
            periodicSwapEvents++;
            swappedByPeriodic += swaps;
        }
        if (listener != null) {
            listener.onSwap(TriggerType.PERIODIC, updates, oldSel, selection.clone(),
                    replacedOut, replacedIn, scores.clone(), tenure.clone(),
                    stabilityRatio(oldSel, selection));
        }
    }

    private void emitPeriodicNoSwap(double[] scores) {
        if (listener == null) return;
        listener.onSwap(TriggerType.PERIODIC, updates, selection, selection.clone(),
                Collections.<Integer>emptySet(), Collections.<Integer>emptySet(),
                scores.clone(), tenure.clone(), 1.0);
    }

    private double stabilityRatio(int[] oldSel, int[] newSel) {
        Set<Integer> oldSet = new HashSet<>();
        for (int i : oldSel) oldSet.add(i);
        int kept = 0;
        for (int i : newSel) if (oldSet.contains(i)) kept++;
        return (double) kept / oldSel.length;
    }

    private static void sortByScoreStable(int[] arr, double[] scores, boolean ascending) {
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

    private void commitSelection(Set<Integer> newSet) {
        if (newSet.size() != k) {
            throw new IllegalStateException("commitSelection: expected " + k +
                    " features, got " + newSet.size());
        }
        int[] arr = new int[k];
        int idx = 0;
        for (int f : newSet) arr[idx++] = f;
        Arrays.sort(arr);
        this.selection = arr;
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
        return "DriftAwareSelector(K=" + k + ", N=" + periodN +
                ", N_min=" + minTenure + ", W_pd=" + wPostDrift +
                ", maxSwap=" + maxSwapsPerCycle +
                ", tieEps=" + tieEpsilon + ", decay=" + driftDecay + ")";
    }

    private void ensureInitialized() {
        if (!initialized) throw new IllegalStateException("selector not initialized");
    }

    private static boolean allFinite(double[] row) {
        for (double v : row) if (!Double.isFinite(v)) return false;
        return true;
    }

    private static class Collections {
        @SuppressWarnings("unchecked")
        static <T> Set<T> emptySet() { return java.util.Collections.emptySet(); }
    }
}