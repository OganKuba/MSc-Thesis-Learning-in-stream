package thesis.selection;

import thesis.discretization.PiDDiscretizer;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.function.BiFunction;

public class DriftAwareSelector implements FeatureSelector {

    private final int numFeatures;
    private final int numClasses;
    private final int k;
    private final int periodN;
    private final int minTenure;
    private final int wPostDrift;
    private final int maxSwapsPerCycle;
    private final PiDDiscretizer discretizer;
    private final BiFunction<Integer, Integer, FilterRanker> rankerFactory;

    private int[] selection;
    private long[] tenure;
    private long updates;

    private int[][] ringBins;
    private int[] ringLabels;
    private int ringPos;
    private int ringCount;

    private boolean collecting;
    private int collected;
    private FilterRanker postDriftRanker;
    private Set<Integer> pendingDriftingFeatures;

    private boolean initialized;
    private long periodicSwapEvents;
    private long alarmSwapEvents;
    private long swappedByAlarm;
    private long swappedByPeriodic;

    public DriftAwareSelector(int numFeatures, int numClasses) {
        this(numFeatures, numClasses,
                StaticFeatureSelector.defaultK(numFeatures),
                1000, 500, 500,
                new PiDDiscretizer(numFeatures, numClasses),
                (bins, classes) -> new InformationGainRanker(numFeatures, bins, classes));
    }

    public DriftAwareSelector(int numFeatures, int numClasses, int k,
                              int periodN, int minTenure, int wPostDrift,
                              PiDDiscretizer discretizer,
                              BiFunction<Integer, Integer, FilterRanker> rankerFactory) {
        if (numFeatures < 1) throw new IllegalArgumentException("numFeatures must be >= 1");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        if (k < 1 || k > numFeatures) {
            throw new IllegalArgumentException("k must be in [1, " + numFeatures + "]");
        }
        if (periodN < 100) throw new IllegalArgumentException("periodN must be >= 100");
        if (minTenure < 0) throw new IllegalArgumentException("minTenure must be >= 0");
        if (wPostDrift < 50) throw new IllegalArgumentException("wPostDrift must be >= 50");
        if (discretizer.getNumFeatures() != numFeatures) {
            throw new IllegalArgumentException("discretizer numFeatures mismatch");
        }
        this.numFeatures = numFeatures;
        this.numClasses = numClasses;
        this.k = k;
        this.periodN = periodN;
        this.minTenure = minTenure;
        this.wPostDrift = wPostDrift;
        this.maxSwapsPerCycle = Math.max(1, (int) Math.ceil(k * 0.3));
        this.discretizer = discretizer;
        this.rankerFactory = rankerFactory;
        this.ringBins = new int[periodN][];
        this.ringLabels = new int[periodN];
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

        FilterRanker ranker = rankerFactory.apply(discretizer.getB2(), numClasses);
        for (int i = 0; i < initialWindow.length; i++) {
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
    }

    @Override
    public void update(double[] instance, int classLabel,
                       boolean driftAlarm, Set<Integer> driftingFeatures) {
        if (!initialized) return;
        if (instance.length != numFeatures) {
            throw new IllegalArgumentException("expected " + numFeatures + " features");
        }

        discretizer.update(instance, classLabel);
        if (!discretizer.isReady()) return;

        int[] bins = discretizer.discretizeAll(instance);
        pushRing(bins, classLabel);
        updates++;
        for (int idx : selection) tenure[idx]++;

        if (driftAlarm && !collecting) {
            startCollecting(driftingFeatures);
        }

        if (collecting) {
            postDriftRanker.update(bins, classLabel);
            collected++;
            if (collected >= wPostDrift) {
                whereSwap();
                collecting = false;
                collected = 0;
                postDriftRanker = null;
                pendingDriftingFeatures = null;
            }
        }

        if (updates % periodN == 0) {
            periodicReSelect();
        }
    }

    private void startCollecting(Set<Integer> driftingFeatures) {
        collecting = true;
        collected = 0;
        postDriftRanker = rankerFactory.apply(discretizer.getB2(), numClasses);
        pendingDriftingFeatures = (driftingFeatures == null)
                ? new HashSet<>() : new HashSet<>(driftingFeatures);
    }

    private void whereSwap() {
        Set<Integer> currentSet = new LinkedHashSet<>();
        for (int idx : selection) currentSet.add(idx);

        Set<Integer> outCandidatesSet = new LinkedHashSet<>();
        for (int idx : selection) {
            if (pendingDriftingFeatures.contains(idx) && tenure[idx] >= minTenure) {
                outCandidatesSet.add(idx);
            }
        }
        if (outCandidatesSet.isEmpty()) return;

        double[] scores = postDriftRanker.getFeatureScores();

        Integer[] outCandidates = outCandidatesSet.stream()
                .sorted(Comparator.comparingDouble(idx -> scores[idx]))
                .toArray(Integer[]::new);

        Integer[] inCandidates = new Integer[numFeatures];
        int j = 0;
        for (int f = 0; f < numFeatures; f++) {
            if (currentSet.contains(f)) continue;
            if (pendingDriftingFeatures.contains(f)) continue;
            inCandidates[j++] = f;
        }
        inCandidates = Arrays.copyOf(inCandidates, j);
        Arrays.sort(inCandidates, Comparator.comparingDouble((Integer idx) -> -scores[idx]));

        Set<Integer> newSet = new LinkedHashSet<>(currentSet);
        int swaps = 0;
        int maxSwaps = Math.min(outCandidates.length, inCandidates.length);
        for (int s = 0; s < maxSwaps; s++) {
            int outIdx = outCandidates[s];
            int inIdx = inCandidates[s];
            if (scores[inIdx] > scores[outIdx]) {
                newSet.remove(outIdx);
                newSet.add(inIdx);
                tenure[inIdx] = 0;
                swaps++;
            } else {
                break;
            }
        }

        if (swaps > 0) {
            commitSelection(newSet);
            alarmSwapEvents++;
            swappedByAlarm += swaps;
        }
    }

    private void periodicReSelect() {
        FilterRanker ranker = rankerFactory.apply(discretizer.getB2(), numClasses);
        for (int i = 0; i < ringCount; i++) {
            ranker.update(ringBins[i], ringLabels[i]);
        }
        double[] scores = ranker.getFeatureScores();

        Set<Integer> currentSet = new HashSet<>();
        for (int idx : selection) currentSet.add(idx);

        Integer[] outCandidates = currentSet.stream()
                .filter(idx -> tenure[idx] >= minTenure)
                .sorted(Comparator.comparingDouble(idx -> scores[idx]))
                .toArray(Integer[]::new);

        Integer[] inCandidates = new Integer[numFeatures - selection.length];
        int j = 0;
        for (int f = 0; f < numFeatures; f++) {
            if (!currentSet.contains(f)) inCandidates[j++] = f;
        }
        Arrays.sort(inCandidates, Comparator.comparingDouble((Integer idx) -> -scores[idx]));

        Set<Integer> newSet = new HashSet<>(currentSet);
        int swaps = 0;
        for (int s = 0; s < maxSwapsPerCycle; s++) {
            if (s >= outCandidates.length || s >= inCandidates.length) break;
            int outIdx = outCandidates[s];
            int inIdx = inCandidates[s];
            if (scores[inIdx] > scores[outIdx]) {
                newSet.remove(outIdx);
                newSet.add(inIdx);
                tenure[inIdx] = 0;
                swaps++;
            } else {
                break;
            }
        }

        if (swaps > 0) {
            commitSelection(newSet);
            periodicSwapEvents++;
            swappedByPeriodic += swaps;
        }
    }

    private void commitSelection(Set<Integer> newSet) {
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

    public boolean isCollecting()         { return collecting; }
    public int getCollectedCount()        { return collected; }
    public int getPeriodN()               { return periodN; }
    public int getMinTenure()             { return minTenure; }
    public int getWPostDrift()            { return wPostDrift; }
    public int getMaxSwapsPerCycle()      { return maxSwapsPerCycle; }
    public long getPeriodicSwapEvents()   { return periodicSwapEvents; }
    public long getAlarmSwapEvents()      { return alarmSwapEvents; }
    public long getSwappedByAlarm()       { return swappedByAlarm; }
    public long getSwappedByPeriodic()    { return swappedByPeriodic; }

    @Override public int getNumFeatures()    { return numFeatures; }
    @Override public int getK()              { return k; }
    @Override public boolean isInitialized() { return initialized; }
    @Override public String name() {
        return "DriftAwareSelector(K=" + k + ", N=" + periodN +
                ", N_min=" + minTenure + ", W_pd=" + wPostDrift +
                ", maxPerSwap=" + maxSwapsPerCycle + ")";
    }

    private void ensureInitialized() {
        if (!initialized) throw new IllegalStateException("selector not initialized");
    }
}