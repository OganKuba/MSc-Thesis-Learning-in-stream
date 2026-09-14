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

    public enum RecoveryOutcome {
        RECOVERED,
        UNRECOVERED,
        NO_DROP,
        CANCELLED
    }

    private final double recoveryTolerance = 0.05;
    private final int recoveryBudgetInstances = 10_000;
    private final int dropGraceInstances;

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
        this.dropGraceInstances = 2 * this.windowSize;
    }

    public void onInitialSelection(long instanceIndex, int[] selection) {
        stability.update(selection == null ? new int[0] : selection);
        recordSelection(instanceIndex, "initial", selection, null);
        if (selection != null) lastSelection = selection.clone();
        this.lastSelectionChangeInstance = instanceIndex;
        this.lastWindowEnd = instanceIndex;
    }

    public void onInstance(long instanceIndex, int yTrue, int yPred,
                           int[] currentSelection, boolean driftAlarm,
                           Set<Integer> driftingFeatures,
                           MetricsCollector metrics) {
        onInstance(instanceIndex, yTrue, yPred, currentSelection, driftAlarm,
                driftingFeatures, metrics, null);
    }

    public void onInstance(long instanceIndex, int yTrue, int yPred,
                           int[] currentSelection, boolean driftAlarm,
                           Set<Integer> driftingFeatures,
                           MetricsCollector metrics,
                           String selectionTrigger) {
        onInstance(instanceIndex, yTrue, yPred, currentSelection, driftAlarm,
                driftingFeatures, metrics, selectionTrigger, true);
    }

    public void onInstance(long instanceIndex, int yTrue, int yPred,
                           int[] currentSelection, boolean driftAlarm,
                           Set<Integer> driftingFeatures,
                           MetricsCollector metrics,
                           String selectionTrigger,
                           boolean selectorDrivesModel) {
        double windowAcc = metrics.getAccuracy().getAccuracy();

        if (driftAlarm) {
            windowDriftCount++;
            totalDriftCount++;
            recordDriftAlarm(instanceIndex, driftingFeatures, yTrue != yPred ? 1.0 : 0.0, windowAcc);
            startRecoveryTracking(instanceIndex);
        } else if (recoveryTracking) {
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
            String trigger;
            if (lastSelectionChangeInstance < 0) {
                trigger = "initial";
            } else if (!selectorDrivesModel) {
                trigger = "subspace_change";
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
        r.perLearnerAction = encodeActions(summary);
        r.perLearnerOverlap = encodeInts(summary == null ? null : summary.getOverlapCounts());
        r.perLearnerSubspace = encodeInts(summary == null ? null : summary.getSubspaceSizes());
        adaptations.add(r);
    }

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
        r.keptCount = promotionDelta;            // intrinsic: background promotions
        r.surgicalCount = extSurgicalDelta;      // A2: external surgical swaps
        r.fullReplacementCount = intrinsicResetDelta;  // intrinsic: full resets (no background)
        r.noReplacementCount = 0;
        r.extKeepCount = extKeepDelta;           // external: learners kept
        r.extFullCount = extFullDelta;           // external: full resets
        adaptations.add(r);
    }

    private void pushAccHistory(double windowAcc) {
        accHistory[accHistoryIdx] = windowAcc;
        accHistoryIdx = (accHistoryIdx + 1) % accHistory.length;
        if (accHistoryCount < accHistory.length) accHistoryCount++;
    }

    private double laggedAccuracy(double fallback) {
        if (accHistoryCount == 0) return fallback;
        double v = (accHistoryCount < accHistory.length)
                ? accHistory[0]                 // ring not yet wrapped: index 0 is the oldest
                : accHistory[accHistoryIdx];    // ring full: the write head points at the oldest
        return Double.isFinite(v) ? v : fallback;
    }

    private void startRecoveryTracking(long instanceIndex) {
        if (recoveryTracking) {
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
            finishRecovery(lastInstance,
                    dropObserved ? RecoveryOutcome.UNRECOVERED : RecoveryOutcome.NO_DROP);
        }
        if (lastInstance > lastWindowEnd) {
            MetricsCollector.Snapshot snap = metrics.snapshot();
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

    public double meanRecoveryLength() {
        long n = 0; double s = 0.0;
        for (RecoveryRow r : recoveries) {
            if (r.outcome == RecoveryOutcome.RECOVERED && r.recoveryLength > 0) {
                s += r.recoveryLength; n++;
            }
        }
        return n == 0 ? Double.NaN : s / n;
    }

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
        public double kappaTemporal;
        public double ramHoursGB;
        public double peakMB;
        public double throughput;
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
        public long dropInstance;
        public long instancesToDrop;
        public long recoveredInstance;
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
        public String perLearnerAction = "";
        public String perLearnerOverlap = "";
        public String perLearnerSubspace = "";
    }
}
