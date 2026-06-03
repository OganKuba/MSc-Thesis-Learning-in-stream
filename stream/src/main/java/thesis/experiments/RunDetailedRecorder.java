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

    private final double recoveryTolerance = 0.05;
    private final int recoveryWindow = 10_000;
    private boolean recoveryTracking;
    private long currentDriftId;
    private long currentDriftInstance;
    private double currentBaselineAcc;
    private double currentThresholdAcc;
    private double currentMaxDrop;
    private double currentRecoveryArea;
    private long currentRecoveryTicks;
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
        double windowAcc = metrics.getAccuracy().getAccuracy();

        if (driftAlarm) {
            windowDriftCount++;
            totalDriftCount++;
            recordDriftAlarm(instanceIndex, driftingFeatures, yTrue != yPred ? 1.0 : 0.0, windowAcc);
            startRecoveryTracking(instanceIndex, windowAcc);
        }

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
            String trigger = driftAlarm ? "drift_alarm"
                    : (lastSelectionChangeInstance < 0 ? "initial" : "selection_change");
            recordSelection(instanceIndex, trigger, currentSelection, driftingFeatures);
            lastSelection = currentSelection.clone();
            lastSelectionChangeInstance = instanceIndex;
        }

        if (instanceIndex - lastWindowEnd >= windowSize) {
            MetricsCollector.Snapshot snap = metrics.snapshot();
            double throughput = snap.avgUpdateMicros > 0.0
                    ? 1_000_000.0 / snap.avgUpdateMicros
                    : 0.0;
            recordWindow(lastWindowEnd + 1, instanceIndex,
                    snap.accuracyWindow, snap.kappa, snap.kappaPer,
                    Double.NaN, // temporal kappa (window-local) — placeholder until exposed
                    snap.ramHoursGB, snap.peakMB, throughput);
            if (recoveryTracking) tickRecoveryWindow(instanceIndex, snap.accuracyWindow);
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
                              double accuracy, double kappa, double kappaPer,
                              double temporalKappa,
                              double ramHoursGB, double peakMB, double throughput) {
        WindowRow r = new WindowRow();
        r.windowId = windowsEmitted++;
        r.startInstance = start;
        r.endInstance = end;
        r.accuracy = accuracy;
        r.kappa = kappa;
        r.kappaPer = kappaPer;
        r.temporalKappa = temporalKappa;
        r.ramHoursGB = ramHoursGB;
        r.peakMB = peakMB;
        r.throughput = throughput;
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
        adaptations.add(r);
    }

    public void onDAARFEvent(long instanceIndex, long extKeepDelta, long extFullDelta) {
        AdaptationRow r = new AdaptationRow();
        r.instanceIndex = instanceIndex;
        r.eventType = "da_arf_ext";
        r.keptCount = 0;
        r.surgicalCount = 0;
        r.fullReplacementCount = 0;
        r.noReplacementCount = 0;
        r.extKeepCount = extKeepDelta;
        r.extFullCount = extFullDelta;
        adaptations.add(r);
    }

    private void startRecoveryTracking(long instanceIndex, double baselineAcc) {
        if (recoveryTracking) {
            finishRecovery(instanceIndex, -1, true);
        }
        driftCounter++;
        currentDriftId = driftCounter;
        currentDriftInstance = instanceIndex;
        currentBaselineAcc = Double.isFinite(baselineAcc) ? baselineAcc : 0.0;
        currentThresholdAcc = currentBaselineAcc - recoveryTolerance;
        currentMaxDrop = 0.0;
        currentRecoveryArea = 0.0;
        currentRecoveryTicks = 0;
        recoveryTracking = true;
    }

    private void tickRecoveryWindow(long instanceIndex, double currentAcc) {
        if (!recoveryTracking) return;
        currentRecoveryTicks++;
        if (Double.isFinite(currentAcc)) {
            double drop = Math.max(0.0, currentBaselineAcc - currentAcc);
            if (drop > currentMaxDrop) currentMaxDrop = drop;
            currentRecoveryArea += drop;
        }
        long elapsedInstances = instanceIndex - currentDriftInstance;
        long elapsedWindows = currentRecoveryTicks;
        if (elapsedWindows >= 1 && Double.isFinite(currentAcc)
                && currentAcc >= currentThresholdAcc) {
            finishRecovery(instanceIndex, elapsedWindows, false);
        } else if (elapsedInstances >= recoveryWindow) {
            finishRecovery(instanceIndex, -1, false);
        }
    }

    private void finishRecovery(long instanceIndex, long recoveryLength, boolean cancelled) {
        RecoveryRow r = new RecoveryRow();
        r.driftId = currentDriftId;
        r.driftInstance = currentDriftInstance;
        if (recoveryLength > 0) {
            r.recoveredInstance = instanceIndex;
            r.recoveryLength = recoveryLength;
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
            finishRecovery(lastInstance, -1, false);
        }
        if (lastInstance > lastWindowEnd) {
            MetricsCollector.Snapshot snap = metrics.snapshot();
            double throughput = snap.avgUpdateMicros > 0.0
                    ? 1_000_000.0 / snap.avgUpdateMicros
                    : 0.0;
            recordWindow(lastWindowEnd + 1, lastInstance,
                    snap.accuracyWindow, snap.kappa, snap.kappaPer,
                    Double.NaN,
                    snap.ramHoursGB, snap.peakMB, throughput);
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
        for (RecoveryRow r : recoveries) if (r.recoveryLength > 0) { s += r.recoveryLength; n++; }
        return n == 0 ? Double.NaN : s / n;
    }

    public long selectionChangeCount() {
        long c = 0;
        for (SelectionRow r : selections) if (!"initial".equals(r.triggerType)) c++;
        return c;
    }

    public double meanTemporalKappa() {
        if (windows.isEmpty()) return Double.NaN;
        double s = 0.0;
        int n = 0;
        for (WindowRow w : windows) if (Double.isFinite(w.kappaPer)) { s += w.kappaPer; n++; }
        return n == 0 ? Double.NaN : s / n;
    }

    public double stdTemporalKappa() {
        if (windows.size() < 2) return 0.0;
        double m = meanTemporalKappa();
        if (Double.isNaN(m)) return 0.0;
        double s = 0.0; int n = 0;
        for (WindowRow w : windows) if (Double.isFinite(w.kappaPer)) { s += (w.kappaPer - m) * (w.kappaPer - m); n++; }
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
        public double kappaPer;
        public double temporalKappa;
        public double ramHoursGB;
        public double peakMB;
        public double throughput;
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
        public long recoveredInstance;
        public long recoveryLength;
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
    }
}
