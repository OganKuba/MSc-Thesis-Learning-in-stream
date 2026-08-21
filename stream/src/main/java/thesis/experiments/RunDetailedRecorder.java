package thesis.experiments;

import thesis.evaluation.FeatureStabilityRatio;
import thesis.evaluation.MetricsCollector;
import thesis.models.DriftActionSummary;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

/**
 * Per-run buffer for the detailed evaluation artifacts that used to live next to
 * {@code thesis.evaluation.*}: window-level metrics, drift alarms, feature
 * selection trajectory, feature importance snapshots, recovery-time records and
 * DA-SRP / DA-ARF adaptation events.
 *
 * <p>One instance is owned by a single worker thread inside
 * {@link UnifiedStreamExperimentRunner#runOne}; the runner drains all recorders
 * single-threaded after the pool finishes, then writes per-block CSVs.
 */
public final class RunDetailedRecorder {

    public final String blockId, dataset, variant, model, selector, detector;
    public final int seed;
    public final int numFeatures;
    public final int windowSize;

    public final List<WindowRow> windows = new ArrayList<>();
    public final List<DriftAlarmRow> driftAlarms = new ArrayList<>();
    public final List<SelectionRow> selections = new ArrayList<>();
    public final List<ImportanceRow> importanceSnapshots = new ArrayList<>();
    public final List<RecoveryRow> recoveries = new ArrayList<>();
    public final List<AdaptationRow> adaptations = new ArrayList<>();

    public long totalDriftCount;
    public long windowsEmitted;

    private long lastWindowEnd;
    private long windowDriftCount;

    private int[] lastSelection;
    private final FeatureStabilityRatio stability = new FeatureStabilityRatio();
    private long lastSelectionChangeInstance = -1;
    private final List<long[]> pendingDriftAfter = new ArrayList<>();

    /**
     * How a recovery episode ended. The previous implementation collapsed all of these into a
     * single {@code recovery_length} of {@code -1}, which made "never recovered" indistinguishable
     * from "superseded by the next alarm" and from "the alarm was never followed by any measurable
     * degradation at all".
     */
    public enum RecoveryOutcome {
        /** Accuracy dropped below the pre-drift threshold and climbed back. */
        RECOVERED,
        /** Accuracy dropped and had still not returned when the budget ran out. */
        UNRECOVERED,
        /** No measurable degradation followed the alarm within the grace period. */
        NO_DROP,
        /** A new alarm arrived while this episode was still open. */
        CANCELLED
    }

    private final double recoveryTolerance = 0.05;
    /** Instances allowed for the accuracy to climb back before the episode is written off. */
    private final int recoveryBudgetInstances = 10_000;
    /** Instances allowed for degradation to appear at all before the episode is NO_DROP. */
    private final int dropGraceInstances;

    /**
     * Per-instance ring of the sliding-window accuracy, one full window long. The baseline for an
     * episode is read from the far end of this ring — i.e. the accuracy as it stood one whole
     * window <i>before</i> the alarm. Taking the baseline at alarm time instead (the old
     * behaviour) sampled an already-degraded value, because a detector only fires once the error
     * has risen and the trailing accuracy window has therefore already absorbed the drift.
     */
    private final double[] accHistory;
    private int accHistoryIdx;
    private int accHistoryCount;

    private boolean recoveryTracking;
    private boolean dropObserved;
    private long currentDriftId;
    private long currentDriftInstance;
    private long currentDropInstance;
    private double currentBaselineAcc;
    private double currentThresholdAcc;
    private double currentMaxDrop;
    private double currentRecoveryArea;
    private long driftCounter;

    public RunDetailedRecorder(String blockId, String dataset, String variant,
                               String model, String selector, String detector,
                               int seed, int numFeatures, int windowSize) {
        this.blockId = blockId;
        this.dataset = dataset;
        this.variant = variant;
        this.model = model;
        this.selector = selector;
        this.detector = detector;
        this.seed = seed;
        this.numFeatures = numFeatures;
        this.windowSize = Math.max(1, windowSize);
        this.accHistory = new double[this.windowSize];
        // Degradation shows up in a trailing window only after that window has refilled with
        // post-drift instances, so allow two windows before calling an episode NO_DROP.
        this.dropGraceInstances = 2 * this.windowSize;
    }

    public void onInitialSelection(long instanceIndex, int[] selection) {
        stability.update(selection == null ? new int[0] : selection);
        recordSelection(instanceIndex, "initial", selection, null);
        if (selection != null) lastSelection = selection.clone();
        // Anchor window boundaries at the post-warmup instance — the first online window
        // starts at (lastWindowEnd + 1) and ends after windowSize prequential steps.
        this.lastWindowEnd = instanceIndex;
    }

    public void onInstance(long instanceIndex, int yTrue, int yPred,
                           int[] currentSelection, boolean driftAlarm,
                           Set<Integer> driftingFeatures,
                           MetricsCollector metrics) {
        onInstance(instanceIndex, yTrue, yPred, currentSelection, driftAlarm,
                driftingFeatures, metrics, null);
    }

    /**
     * @param selectionTrigger what the selector says caused its latest change
     *                         ({@link thesis.selection.FeatureSelector#lastSelectionTrigger()}).
     *                         Passing {@code null} falls back to the old, uninformative
     *                         "selection_change" label.
     */
    public void onInstance(long instanceIndex, int yTrue, int yPred,
                           int[] currentSelection, boolean driftAlarm,
                           Set<Integer> driftingFeatures,
                           MetricsCollector metrics,
                           String selectionTrigger) {
        double windowAcc = metrics.getAccuracy().getAccuracy();

        if (driftAlarm) {
            windowDriftCount++;
            totalDriftCount++;
            recordDriftAlarm(instanceIndex, driftingFeatures, yTrue != yPred ? 1.0 : 0.0, windowAcc);
            startRecoveryTracking(instanceIndex);
        } else if (recoveryTracking) {
            // Evaluated every instance, not once per window: the old code only looked at window
            // boundaries, so the shortest representable recovery was "1 window" and the metric
            // could take barely a handful of distinct values.
            trackRecovery(instanceIndex, windowAcc);
        }
        pushAccHistory(windowAcc);

        if (!pendingDriftAfter.isEmpty()) {
            Iterator<long[]> it = pendingDriftAfter.iterator();
            while (it.hasNext()) {
                long[] e = it.next();
                if (instanceIndex >= e[1]) {
                    driftAlarms.get((int) e[0]).windowAccuracyAfter = windowAcc;
                    it.remove();
                }
            }
        }

        if (currentSelection != null && !sameSelection(currentSelection, lastSelection)) {
            stability.update(currentSelection);
            // Ask the selector rather than guessing from the alarm flag: an alarm-driven re-rank
            // lands wPostDrift instances AFTER the alarm, when driftAlarm is already false.
            String trigger;
            if (lastSelectionChangeInstance < 0) {
                trigger = "initial";
            } else if (selectionTrigger != null && !selectionTrigger.isEmpty()) {
                trigger = selectionTrigger;
            } else {
                trigger = driftAlarm ? "drift_alarm" : "selection_change";
            }
            recordSelection(instanceIndex, trigger, currentSelection, driftingFeatures);
            lastSelection = currentSelection.clone();
            lastSelectionChangeInstance = instanceIndex;
        }

        if (instanceIndex - lastWindowEnd >= windowSize) {
            MetricsCollector.Snapshot snap = metrics.snapshot();
            // Throughput from the FULL step, not from predict alone. The old formula answered
            // "how fast could this model predict if it never learned", producing up to 39.7 M
            // instances/s and disagreeing with the run-level throughput in master_summary.csv.
            double throughput = snap.avgStepMicros > 0.0
                    ? 1_000_000.0 / snap.avgStepMicros
                    : 0.0;
            recordWindow(lastWindowEnd + 1, instanceIndex,
                    snap.accuracyWindow, snap.kappa, snap.kappaTemporal,
                    snap.ramHoursGB, snap.peakMB, throughput, snap.avgPredictMicros);
            lastWindowEnd = instanceIndex;
            windowDriftCount = 0;
        }
    }

    private boolean sameSelection(int[] a, int[] b) {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        if (a.length != b.length) return false;
        int[] aa = a.clone();
        int[] bb = b.clone();
        Arrays.sort(aa);
        Arrays.sort(bb);
        return Arrays.equals(aa, bb);
    }

    private void recordWindow(long start, long end,
                              double accuracy, double kappa, double kappaTemporal,
                              double ramHoursGB, double peakMB, double throughput,
                              double predictLatencyMicros) {
        WindowRow r = new WindowRow();
        r.windowId = windowsEmitted++;
        r.startInstance = start;
        r.endInstance = end;
        r.accuracy = accuracy;
        r.kappa = kappa;
        r.kappaTemporal = kappaTemporal;
        r.ramHoursGB = ramHoursGB;
        r.peakMB = peakMB;
        r.throughput = throughput;
        r.predictLatencyMicros = predictLatencyMicros;
        r.driftCountInWindow = windowDriftCount;
        r.totalDriftCount = totalDriftCount;
        windows.add(r);
    }

    private void recordDriftAlarm(long instanceIndex, Set<Integer> drifting,
                                  double error, double windowAccuracyBefore) {
        DriftAlarmRow r = new DriftAlarmRow();
        r.instanceIndex = instanceIndex;
        r.globalAlarm = true;
        r.driftingFeatures = drifting == null || drifting.isEmpty()
                ? new int[0] : toSortedArray(drifting);
        r.numDriftingFeatures = r.driftingFeatures.length;
        r.errorAtAlarm = error;
        r.windowAccuracyBefore = windowAccuracyBefore;
        r.windowAccuracyAfter = Double.NaN;
        driftAlarms.add(r);
        pendingDriftAfter.add(new long[]{driftAlarms.size() - 1, instanceIndex + windowSize});
    }

    private void recordSelection(long instanceIndex, String trigger,
                                 int[] selection, Set<Integer> drifting) {
        SelectionRow r = new SelectionRow();
        r.instanceIndex = instanceIndex;
        r.triggerType = trigger;
        r.selectedFeatures = selection == null ? new int[0] : selection.clone();
        Arrays.sort(r.selectedFeatures);
        r.selectedFeatureCount = r.selectedFeatures.length;
        if (lastSelection != null) {
            int[] changed = diff(lastSelection, r.selectedFeatures);
            r.changedFeatures = changed;
            r.jaccardToPrevious = jaccard(lastSelection, r.selectedFeatures);
        } else {
            r.changedFeatures = new int[0];
            r.jaccardToPrevious = Double.NaN;
        }
        double avgStab = stability.getAverageRatio();
        r.stabilityRatio = Double.isNaN(avgStab) ? 1.0 : avgStab;
        selections.add(r);
    }

    public void onImportanceSnapshot(long instanceIndex, double[] importance,
                                     int[] selection, Set<Integer> drifting) {
        if (importance == null) return;
        BitSet selSet = new BitSet(numFeatures);
        if (selection != null) for (int s : selection) if (s >= 0 && s < numFeatures) selSet.set(s);
        BitSet driftSet = new BitSet(numFeatures);
        if (drifting != null) for (Integer d : drifting) if (d != null && d >= 0 && d < numFeatures) driftSet.set(d);
        Integer[] order = new Integer[importance.length];
        for (int i = 0; i < importance.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(importance[b], importance[a]));
        int[] rankOf = new int[importance.length];
        for (int r = 0; r < order.length; r++) rankOf[order[r]] = r;
        for (int i = 0; i < importance.length; i++) {
            ImportanceRow row = new ImportanceRow();
            row.instanceIndex = instanceIndex;
            row.featureIndex = i;
            row.importance = importance[i];
            row.rank = rankOf[i];
            row.isSelected = selSet.get(i);
            row.isDrifting = driftSet.get(i);
            importanceSnapshots.add(row);
        }
    }

    public void onDASRPEvent(long instanceIndex, DriftActionSummary summary) {
        AdaptationRow r = new AdaptationRow();
        r.instanceIndex = instanceIndex;
        r.eventType = "da_drift";
        r.keptCount = summary == null ? 0 : summary.getKeptCount();
        r.surgicalCount = summary == null ? 0 : summary.getSurgicalCount();
        r.fullReplacementCount = summary == null ? 0 : summary.getFullCount();
        r.noReplacementCount = summary == null ? 0 : summary.getNoReplacementCount();
        r.extKeepCount = 0;
        r.extFullCount = 0;
        // The four counters above are the ensemble-level view. DriftActionSummary already
        // carries the per-learner detail — which learner did what, and how many of the
        // drifting features were inside its subspace — and collapsing it here was what made
        // "which learners does a surgical swap actually hit" unanswerable from the CSVs.
        // Kept as three pipe-encoded columns (one entry per learner, ensemble order) so the
        // row count is unchanged and DA-ARF rows simply leave them empty.
        r.perLearnerAction = encodeActions(summary);
        r.perLearnerOverlap = encodeInts(summary == null ? null : summary.getOverlapCounts());
        r.perLearnerSubspace = encodeInts(summary == null ? null : summary.getSubspaceSizes());
        adaptations.add(r);
    }

    /** One letter per learner: K=KEEP, S=SURGICAL, F=FULL, N=NO_REPLACEMENT. */
    private static String encodeActions(DriftActionSummary s) {
        if (s == null) return "";
        DriftActionSummary.Action[] actions = s.getPerLearner();
        StringBuilder sb = new StringBuilder(actions.length * 2);
        for (int i = 0; i < actions.length; i++) {
            if (i > 0) sb.append('|');
            switch (actions[i]) {
                case KEEP:           sb.append('K'); break;
                case SURGICAL:       sb.append('S'); break;
                case FULL:           sb.append('F'); break;
                case NO_REPLACEMENT: sb.append('N'); break;
            }
        }
        return sb.toString();
    }

    private static String encodeInts(int[] values) {
        if (values == null || values.length == 0) return "";
        StringBuilder sb = new StringBuilder(values.length * 3);
        for (int i = 0; i < values.length; i++) {
            if (i > 0) sb.append('|');
            sb.append(values[i]);
        }
        return sb.toString();
    }

    public void onDAARFEvent(long instanceIndex, long extKeepDelta, long extFullDelta,
                             long extSurgicalDelta, long intrinsicResetDelta,
                             long promotionDelta) {
        AdaptationRow r = new AdaptationRow();
        r.instanceIndex = instanceIndex;
        r.eventType = "da_arf";
        // A1: intrinsic resets go to full_replacement, external resets to ext_full,
        // so the two reset channels stay separable in adaptation_events.csv.
        r.keptCount = promotionDelta;            // intrinsic: background promotions
        r.surgicalCount = extSurgicalDelta;      // A2: external surgical swaps
        r.fullReplacementCount = intrinsicResetDelta;  // intrinsic: full resets (no background)
        r.noReplacementCount = 0;
        r.extKeepCount = extKeepDelta;           // external: learners kept
        r.extFullCount = extFullDelta;           // external: full resets
        adaptations.add(r);
    }

    /** Store this instance's sliding-window accuracy so later alarms can look back one window. */
    private void pushAccHistory(double windowAcc) {
        accHistory[accHistoryIdx] = windowAcc;
        accHistoryIdx = (accHistoryIdx + 1) % accHistory.length;
        if (accHistoryCount < accHistory.length) accHistoryCount++;
    }

    /**
     * Accuracy as of one full window ago — the pre-drift baseline. Falls back to the oldest value
     * available while the ring is still filling (early in the run).
     */
    private double laggedAccuracy(double fallback) {
        if (accHistoryCount == 0) return fallback;
        double v = (accHistoryCount < accHistory.length)
                ? accHistory[0]                 // ring not yet wrapped: index 0 is the oldest
                : accHistory[accHistoryIdx];    // ring full: the write head points at the oldest
        return Double.isFinite(v) ? v : fallback;
    }

    private void startRecoveryTracking(long instanceIndex) {
        if (recoveryTracking) {
            // Superseded by a fresh alarm — recorded as CANCELLED so it is not silently pooled
            // with genuine "never recovered" episodes.
            finishRecovery(instanceIndex, RecoveryOutcome.CANCELLED);
        }
        double baseline = laggedAccuracy(0.0);
        driftCounter++;
        currentDriftId = driftCounter;
        currentDriftInstance = instanceIndex;
        currentDropInstance = -1;
        currentBaselineAcc = Double.isFinite(baseline) ? baseline : 0.0;
        currentThresholdAcc = currentBaselineAcc - recoveryTolerance;
        currentMaxDrop = 0.0;
        currentRecoveryArea = 0.0;
        dropObserved = false;
        recoveryTracking = true;
    }

    /**
     * Two-phase episode, evaluated once per instance.
     *
     * <p>Phase 1 waits for accuracy to actually fall below {@code baseline - tolerance}. Without
     * this phase an episode "recovers" immediately, because right after an alarm the trailing
     * accuracy window is still dominated by pre-drift instances and therefore still reads high —
     * that single flaw produced {@code recovery_length == 1} for 64–90 % of all episodes.
     *
     * <p>Phase 2 then measures how long the accuracy takes to climb back to the threshold.
     */
    private void trackRecovery(long instanceIndex, double currentAcc) {
        if (!recoveryTracking) return;
        if (Double.isFinite(currentAcc)) {
            double drop = Math.max(0.0, currentBaselineAcc - currentAcc);
            if (drop > currentMaxDrop) currentMaxDrop = drop;
            currentRecoveryArea += drop;
        }
        long elapsed = instanceIndex - currentDriftInstance;

        if (!dropObserved) {
            if (Double.isFinite(currentAcc) && currentAcc < currentThresholdAcc) {
                dropObserved = true;
                currentDropInstance = instanceIndex;
            } else if (elapsed >= dropGraceInstances) {
                finishRecovery(instanceIndex, RecoveryOutcome.NO_DROP);
            }
            return;
        }

        if (Double.isFinite(currentAcc) && currentAcc >= currentThresholdAcc) {
            finishRecovery(instanceIndex, RecoveryOutcome.RECOVERED);
        } else if (elapsed >= recoveryBudgetInstances) {
            finishRecovery(instanceIndex, RecoveryOutcome.UNRECOVERED);
        }
    }

    private void finishRecovery(long instanceIndex, RecoveryOutcome outcome) {
        RecoveryRow r = new RecoveryRow();
        r.driftId = currentDriftId;
        r.driftInstance = currentDriftInstance;
        r.outcome = outcome;
        r.dropInstance = currentDropInstance;
        r.instancesToDrop = (currentDropInstance < 0) ? -1 : currentDropInstance - currentDriftInstance;
        if (outcome == RecoveryOutcome.RECOVERED) {
            r.recoveredInstance = instanceIndex;
            // Instances from the alarm to the moment accuracy is back at the pre-drift level.
            r.recoveryLength = instanceIndex - currentDriftInstance;
        } else {
            r.recoveredInstance = -1;
            r.recoveryLength = -1;
        }
        r.baselineAccuracyBeforeDrift = currentBaselineAcc;
        r.thresholdAccuracy = currentThresholdAcc;
        r.maxDrop = currentMaxDrop;
        r.areaUnderRecoveryCurve = currentRecoveryArea;
        recoveries.add(r);
        recoveryTracking = false;
    }

    public void finalizeAtEnd(long lastInstance, MetricsCollector metrics) {
        double windowAcc = metrics.getAccuracy().getAccuracy();
        if (recoveryTracking) {
            // The stream ended mid-episode: report what the episode had actually reached, rather
            // than blanket-labelling it "never recovered".
            finishRecovery(lastInstance,
                    dropObserved ? RecoveryOutcome.UNRECOVERED : RecoveryOutcome.NO_DROP);
        }
        if (lastInstance > lastWindowEnd) {
            MetricsCollector.Snapshot snap = metrics.snapshot();
            // Throughput from the FULL step, not from predict alone. The old formula answered
            // "how fast could this model predict if it never learned", producing up to 39.7 M
            // instances/s and disagreeing with the run-level throughput in master_summary.csv.
            double throughput = snap.avgStepMicros > 0.0
                    ? 1_000_000.0 / snap.avgStepMicros
                    : 0.0;
            recordWindow(lastWindowEnd + 1, lastInstance,
                    snap.accuracyWindow, snap.kappa, snap.kappaTemporal,
                    snap.ramHoursGB, snap.peakMB, throughput, snap.avgPredictMicros);
            lastWindowEnd = lastInstance;
        }
        for (long[] e : pendingDriftAfter) {
            DriftAlarmRow row = driftAlarms.get((int) e[0]);
            if (Double.isNaN(row.windowAccuracyAfter)) {
                row.windowAccuracyAfter = windowAcc;
            }
        }
        pendingDriftAfter.clear();
    }

    private static int[] toSortedArray(Set<Integer> s) {
        int[] out = new int[s.size()];
        int i = 0;
        for (Integer v : s) out[i++] = v == null ? -1 : v;
        Arrays.sort(out);
        return out;
    }

    private static int[] diff(int[] prev, int[] curr) {
        BitSet pa = new BitSet();
        BitSet pb = new BitSet();
        for (int v : prev) if (v >= 0) pa.set(v);
        for (int v : curr) if (v >= 0) pb.set(v);
        BitSet sym = (BitSet) pa.clone();
        sym.xor(pb);
        int[] out = new int[sym.cardinality()];
        int j = 0;
        for (int i = sym.nextSetBit(0); i >= 0; i = sym.nextSetBit(i + 1)) out[j++] = i;
        return out;
    }

    private static double jaccard(int[] a, int[] b) {
        if ((a == null || a.length == 0) && (b == null || b.length == 0)) return 1.0;
        BitSet pa = new BitSet();
        BitSet pb = new BitSet();
        if (a != null) for (int v : a) if (v >= 0) pa.set(v);
        if (b != null) for (int v : b) if (v >= 0) pb.set(v);
        BitSet inter = (BitSet) pa.clone(); inter.and(pb);
        BitSet union = (BitSet) pa.clone(); union.or(pb);
        return union.cardinality() == 0 ? 1.0 : (double) inter.cardinality() / union.cardinality();
    }

    public double meanSelectedFeatureCount() {
        if (selections.isEmpty()) return Double.NaN;
        double s = 0.0;
        for (SelectionRow r : selections) s += r.selectedFeatureCount;
        return s / selections.size();
    }

    /** Mean recovery length in INSTANCES over episodes that actually recovered. */
    public double meanRecoveryLength() {
        long n = 0; double s = 0.0;
        for (RecoveryRow r : recoveries) {
            if (r.outcome == RecoveryOutcome.RECOVERED && r.recoveryLength > 0) {
                s += r.recoveryLength; n++;
            }
        }
        return n == 0 ? Double.NaN : s / n;
    }

    /**
     * Mean depth of the post-alarm accuracy dip across <b>all</b> episodes (lower is better).
     *
     * <p>NO_DROP episodes are included with their near-zero drop on purpose: a method that
     * absorbs a drift without degrading deserves the best possible score. That also makes this
     * metric defined for every variant that saw at least one alarm, unlike recovery length, which
     * is undefined whenever nothing ever recovered.
     */
    public double meanMaxDrop() {
        if (recoveries.isEmpty()) return Double.NaN;
        double s = 0.0;
        for (RecoveryRow r : recoveries) s += r.maxDrop;
        return s / recoveries.size();
    }

    public long recoveryOutcomeCount(RecoveryOutcome outcome) {
        long n = 0;
        for (RecoveryRow r : recoveries) if (r.outcome == outcome) n++;
        return n;
    }

    public long selectionChangeCount() {
        long c = 0;
        for (SelectionRow r : selections) if (!"initial".equals(r.triggerType)) c++;
        return c;
    }

    /**
     * Temporal kappa averaged over every window of the run.
     *
     * <p>Named for its aggregation on purpose. The run also reports the temporal kappa of the
     * <i>final</i> window ({@code kappa_temporal_final}); the two used to be published as
     * "temporal_kappa" and "kappa_per", which read as two different metrics even though both are
     * {@link thesis.evaluation.TemporalKappa} — only the aggregation differed.
     */
    public double meanWindowedTemporalKappa() {
        if (windows.isEmpty()) return Double.NaN;
        double s = 0.0;
        int n = 0;
        for (WindowRow w : windows) if (Double.isFinite(w.kappaTemporal)) { s += w.kappaTemporal; n++; }
        return n == 0 ? Double.NaN : s / n;
    }

    public double stdWindowedTemporalKappa() {
        if (windows.size() < 2) return 0.0;
        double m = meanWindowedTemporalKappa();
        if (Double.isNaN(m)) return 0.0;
        double s = 0.0; int n = 0;
        for (WindowRow w : windows) {
            if (Double.isFinite(w.kappaTemporal)) { s += (w.kappaTemporal - m) * (w.kappaTemporal - m); n++; }
        }
        return n < 2 ? 0.0 : Math.sqrt(s / (n - 1));
    }

    public double averageStabilityRatio() {
        double v = stability.getAverageRatio();
        return Double.isNaN(v) ? 1.0 : v;
    }

    public List<RecoveryRow> recoveriesView()       { return Collections.unmodifiableList(recoveries); }
    public List<SelectionRow> selectionsView()      { return Collections.unmodifiableList(selections); }
    public List<DriftAlarmRow> driftAlarmsView()    { return Collections.unmodifiableList(driftAlarms); }

    public static final class WindowRow {
        public long windowId;
        public long startInstance;
        public long endInstance;
        public double accuracy;
        public double kappa;
        /**
         * Temporal kappa (κ_per / κ_+) as it stood at the end of this window. Because the
         * collector's temporal-kappa window and the reporting window are both {@code windowSize}
         * long and their boundaries align, this is the window-local value.
         *
         * <p>There used to be a second field here, {@code temporalKappa}, written as a literal
         * {@code Double.NaN} placeholder "until exposed" — it reached the CSV as a column that was
         * NaN in 100 % of rows while duplicating the name of the metric already stored here.
         */
        public double kappaTemporal;
        public double ramHoursGB;
        public double peakMB;
        /** Instances per second for the complete prequential step (predict + detect + select + train). */
        public double throughput;
        /** Mean {@code model.predict()} latency in microseconds — inference cost on its own. */
        public double predictLatencyMicros;
        public long driftCountInWindow;
        public long totalDriftCount;
    }

    public static final class DriftAlarmRow {
        public long instanceIndex;
        public boolean globalAlarm;
        public int[] driftingFeatures;
        public int numDriftingFeatures;
        public double errorAtAlarm;
        public double windowAccuracyBefore;
        public double windowAccuracyAfter;
    }

    public static final class SelectionRow {
        public long instanceIndex;
        public String triggerType;
        public int[] selectedFeatures;
        public int selectedFeatureCount;
        public int[] changedFeatures;
        public double jaccardToPrevious;
        public double stabilityRatio;
    }

    public static final class ImportanceRow {
        public long instanceIndex;
        public int featureIndex;
        public double importance;
        public int rank;
        public boolean isSelected;
        public boolean isDrifting;
    }

    public static final class RecoveryRow {
        public long driftId;
        public long driftInstance;
        /** Instance at which accuracy first fell below the threshold, or -1 if it never did. */
        public long dropInstance;
        /** Instances from the alarm to the observed drop, or -1 for NO_DROP episodes. */
        public long instancesToDrop;
        public long recoveredInstance;
        /** Instances from alarm to full recovery; -1 unless {@code outcome == RECOVERED}. */
        public long recoveryLength;
        public RecoveryOutcome outcome = RecoveryOutcome.NO_DROP;
        public double baselineAccuracyBeforeDrift;
        public double thresholdAccuracy;
        public double maxDrop;
        public double areaUnderRecoveryCurve;
    }

    public static final class AdaptationRow {
        public long instanceIndex;
        public String eventType;
        public long keptCount;
        public long surgicalCount;
        public long fullReplacementCount;
        public long noReplacementCount;
        public long extKeepCount;
        public long extFullCount;
        /** Pipe-encoded per-learner detail; empty for DA-ARF events. See onDASRPEvent. */
        public String perLearnerAction = "";
        public String perLearnerOverlap = "";
        public String perLearnerSubspace = "";
    }
}
