package thesis.pipeline;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import lombok.Getter;
import moa.streams.InstanceStream;
import thesis.detection.TwoLevelDriftDetector;
import thesis.discretization.PiDDiscretizer;
import thesis.models.DriftAwareSRP;
import thesis.models.FeatureImportance;
import thesis.models.FeatureSpace;
import thesis.models.ModelWrapper;
import thesis.selection.FeatureSelector;
import thesis.selection.FilterRanker;

import java.util.Arrays;
import java.util.Set;

@Getter
public class StreamPipeline {

    public enum TriggerType { NONE, PERIODIC, ALARM, WHERE }

    private final InstanceStream source;
    private final FeatureSelector selector;
    private final ModelWrapper model;
    private final TwoLevelDriftDetector detector;
    private final StreamMetrics metrics;
    private final FeatureSpace space;

    private final PiDDiscretizer rankingPid;
    private final FilterRanker fullRanker;
    private final FeatureImportance importance;

    private final int warmupSize;
    private final int logEvery;
    private final int refreshEvery;
    private final long maxInstances;
    private final boolean verbose;
    private final boolean strict;

    private boolean warmedUp;
    private long processed;
    private long globalAlarmsSeen;
    private long warningsSeen;
    private long periodicTriggers;
    private long whereTriggers;
    private int[] lastSelection;
    private TriggerType lastTrigger = TriggerType.NONE;
    private boolean lastSelectionChanged;

    private StreamPipeline(Builder b) {
        this.source = b.source;
        this.selector = b.selector;
        this.model = b.model;
        this.detector = b.detector;
        this.metrics = (b.metrics != null) ? b.metrics : new StreamMetrics();
        this.rankingPid = b.rankingPid;
        this.fullRanker = b.fullRanker;
        this.importance = b.importance;
        this.warmupSize = b.warmupSize;
        this.logEvery = b.logEvery;
        this.refreshEvery = b.refreshEvery;
        this.maxInstances = b.maxInstances;
        this.verbose = b.verbose;
        this.strict = b.strict;

        if (this.source instanceof moa.options.OptionHandler) {
            ((moa.options.OptionHandler) this.source).prepareForUse();
        }
        InstancesHeader header = source.getHeader();
        if (header == null) {
            throw new IllegalStateException("source stream has no header");
        }
        this.space = new FeatureSpace(header);
        this.warmedUp = selector.isInitialized();

        if (model instanceof DriftAwareSRP) {
            DriftAwareSRP da = (DriftAwareSRP) model;
            da.setScoreProvider(() -> {
                if (fullRanker != null) {
                    double[] s = fullRanker.getFeatureScores();
                    if (s != null && s.length == space.numFeatures()) return s;
                }
                if (importance != null) return importance.getMIScores();
                return new double[space.numFeatures()];
            });
        }

        if (metrics instanceof RecordingMetrics) {
            ((RecordingMetrics) metrics).bindPipeline(this, detector);
        }
    }

    public static Builder builder() { return new Builder(); }

    public void run() {
        warmupIfNeeded();
        while (source.hasMoreInstances() && processed < maxInstances) {
            Instance x = source.nextInstance().getData();
            processInstance(x);
        }
        if (metrics instanceof RecordingMetrics) {
            ((RecordingMetrics) metrics).flushFinalRow();
        }
        if (verbose) finalLog();
    }

    public void processInstance(Instance raw) {
        if (!warmedUp) {
            throw new IllegalStateException("pipeline not warmed up");
        }

        long t0 = System.nanoTime();
        double[] x = space.extractFeatures(raw);
        int y = (int) raw.classValue();

        if (rankingPid != null) {
            rankingPid.update(x, y);
            if (fullRanker != null && rankingPid.isReady()) {
                fullRanker.update(rankingPid.discretizeAll(x), y);
            }
        }

        int yhat = model.predict(raw);
        double error = (yhat == y) ? 0.0 : 1.0;

        detector.update(error, x);
        boolean alarm = detector.isGlobalDriftDetected();
        boolean warning = detectorWarning();
        Set<Integer> drifting = alarm ? detector.getDriftingFeatureIndices() : Set.of();

        if (alarm) {
            globalAlarmsSeen++;
            metrics.onDriftAlarm();
            if (rankingPid != null) {
                for (int idx : drifting) rankingPid.resetFeature(idx);
            }
            if (importance != null && fullRanker != null) {
                double[] sc = fullRanker.getFeatureScores();
                double[] pv = invertPValues(detector.getLastPValues());
                if (sc.length == space.numFeatures() && pv.length == space.numFeatures()) {
                    importance.update(sc, pv);
                } else if (strict) {
                    throw new IllegalStateException(
                            "score/pvalue length mismatch: " + sc.length + "/" + pv.length
                                    + " expected " + space.numFeatures());
                }
            }
        }
        if (warning) warningsSeen++;

        selector.update(x, y, alarm, drifting);
        model.train(raw, y, alarm, drifting);

        TriggerType trig = TriggerType.NONE;
        if (alarm) {
            trig = (model instanceof DriftAwareSRP) ? TriggerType.WHERE : TriggerType.ALARM;
            if (model instanceof DriftAwareSRP) whereTriggers++;
        }

        if (refreshEvery > 0 && processed > 0 && processed % refreshEvery == 0) {
            if (model instanceof DriftAwareSRP) {
                ((DriftAwareSRP) model).refreshAllSubspaces();
            }
            if (trig == TriggerType.NONE) trig = TriggerType.PERIODIC;
            periodicTriggers++;
        }
        lastTrigger = trig;

        int[] currSel = selector.getCurrentSelection();
        boolean changed = lastSelection != null && currSel != null
                && !Arrays.equals(currSel, lastSelection);
        if (changed || lastSelection == null) {
            if (currSel != null) lastSelection = currSel.clone();
        }
        lastSelectionChanged = changed;
        metrics.onSelectionChanged(currSel);

        if (strict && currSel != null && importance != null
                && currSel.length > importance.getNumFeatures()) {
            throw new IllegalStateException("selection size > importance dim");
        }

        long elapsed = System.nanoTime() - t0;
        metrics.update(y, yhat, elapsed);
        processed++;

        if (verbose && logEvery > 0 && processed % logEvery == 0) log();
    }

    private boolean detectorWarning() {
        try {
            java.lang.reflect.Method m = detector.getClass().getMethod("isWarning");
            Object v = m.invoke(detector);
            return v instanceof Boolean && (Boolean) v;
        } catch (Exception ignored) { return false; }
    }

    public void warmupIfNeeded() {
        if (warmedUp) return;
        if (selector.isInitialized()) { warmedUp = true; return; }

        double[][] window = new double[warmupSize][];
        int[] labels = new int[warmupSize];
        int collected = 0;
        while (collected < warmupSize && source.hasMoreInstances()) {
            Instance x = source.nextInstance().getData();
            window[collected] = space.extractFeatures(x);
            labels[collected] = (int) x.classValue();
            if (rankingPid != null) rankingPid.update(window[collected], labels[collected]);
            collected++;
        }
        if (collected == 0) throw new IllegalStateException("no instances during warmup");
        if (collected < warmupSize) {
            window = Arrays.copyOf(window, collected);
            labels = Arrays.copyOf(labels, collected);
        }

        selector.initialize(window, labels);

        if (fullRanker != null && rankingPid != null && rankingPid.isReady()) {
            for (int i = 0; i < window.length; i++) {
                fullRanker.update(rankingPid.discretizeAll(window[i]), labels[i]);
            }
        }
        if (importance != null && fullRanker != null) {
            double[] sc = fullRanker.getFeatureScores();
            if (sc.length == space.numFeatures()) {
                importance.update(sc, new double[space.numFeatures()]);
            }
        }

        warmedUp = true;
        lastSelection = selector.getCurrentSelection() == null
                ? null : selector.getCurrentSelection().clone();
        if (verbose) {
            System.out.printf("[warmup] collected=%d  selector=%s  initial selection=%s%n",
                    collected, selector.name(),
                    Arrays.toString(selector.getCurrentSelection()));
        }
    }

    public TriggerType getLastTrigger() { return lastTrigger; }
    public boolean wasLastSelectionChange() { return lastSelectionChanged; }
    public double[] getLastFeatureScores() {
        return fullRanker == null ? new double[0] : fullRanker.getFeatureScores();
    }

    private void log() {
        Runtime r = Runtime.getRuntime();
        long used = r.totalMemory() - r.freeMemory();
        metrics.recordMemory(used);
        System.out.printf("[t=%6d] acc=%.4f  kappa=%.4f  avg=%.1fµs  alarms=%d  warn=%d  sel=%s  mem=%dMB%n",
                processed, metrics.getAccuracy(), metrics.getKappa(), metrics.getAvgTimeMicros(),
                globalAlarmsSeen, warningsSeen,
                Arrays.toString(selector.getCurrentSelection()),
                used / (1024 * 1024));
    }

    private void finalLog() {
        Runtime r = Runtime.getRuntime();
        long used = r.totalMemory() - r.freeMemory();
        metrics.recordMemory(used);
        System.out.println("─".repeat(80));
        System.out.println("FINAL");
        System.out.printf("  processed=%d  acc=%.4f  kappa=%.4f  avgTime=%.2fµs  alarms=%d  warn=%d  periodic=%d  where=%d%n",
                processed, metrics.getAccuracy(), metrics.getKappa(),
                metrics.getAvgTimeMicros(), globalAlarmsSeen, warningsSeen,
                periodicTriggers, whereTriggers);
        System.out.printf("  peakMem=%dMB  finalSelection=%s%n",
                metrics.getPeakMB(),
                Arrays.toString(selector.getCurrentSelection()));
        System.out.println("  model: " + model.name());
        if (model instanceof DriftAwareSRP) {
            DriftAwareSRP da = (DriftAwareSRP) model;
            System.out.printf("  DA-SRP: handleDrift=%d auto=%d keep=%d surgical=%d full=%d noRepl=%d refresh=%d/%d weighted/fallback=%d/%d%n",
                    da.getHandleDriftCalls(), da.getAutoHandleDriftCalls(),
                    da.getTotalKept(), da.getTotalSurgical(),
                    da.getTotalFull(), da.getTotalNoReplacement(),
                    da.getTotalRefreshed(), da.getRefreshCalls(),
                    da.getWeightedPredictions(), da.getUnweightedFallbacks());
        }
    }

    private static double[] invertPValues(double[] p) {
        double[] out = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            double v = p[i];
            if (Double.isNaN(v) || v < 0.0) v = 0.0;
            if (v > 1.0) v = 1.0;
            out[i] = 1.0 - v;
        }
        return out;
    }

    public static final class Builder {
        private InstanceStream source;
        private FeatureSelector selector;
        private ModelWrapper model;
        private TwoLevelDriftDetector detector;
        private StreamMetrics metrics;
        private PiDDiscretizer rankingPid;
        private FilterRanker fullRanker;
        private FeatureImportance importance;
        private int warmupSize = 1500;
        private int logEvery = 1000;
        private int refreshEvery = 0;
        private long maxInstances = Long.MAX_VALUE;
        private boolean verbose = true;
        private boolean strict = false;

        public Builder source(InstanceStream s)            { this.source = s; return this; }
        public Builder selector(FeatureSelector s)         { this.selector = s; return this; }
        public Builder model(ModelWrapper m)               { this.model = m; return this; }
        public Builder detector(TwoLevelDriftDetector d)   { this.detector = d; return this; }
        public Builder metrics(StreamMetrics m)            { this.metrics = m; return this; }
        public Builder rankingPid(PiDDiscretizer p)        { this.rankingPid = p; return this; }
        public Builder fullRanker(FilterRanker r)          { this.fullRanker = r; return this; }
        public Builder importance(FeatureImportance i)     { this.importance = i; return this; }
        public Builder warmup(int n)                       { this.warmupSize = n; return this; }
        public Builder logEvery(int n)                     { this.logEvery = n; return this; }
        public Builder refreshEvery(int n)                 { this.refreshEvery = n; return this; }
        public Builder maxInstances(long n)                { this.maxInstances = n; return this; }
        public Builder verbose(boolean v)                  { this.verbose = v; return this; }
        public Builder strict(boolean v)                   { this.strict = v; return this; }

        public StreamPipeline build() {
            if (source == null)   throw new IllegalStateException("source must be set");
            if (selector == null) throw new IllegalStateException("selector must be set");
            if (model == null)    throw new IllegalStateException("model must be set");
            if (detector == null) throw new IllegalStateException("detector must be set");
            if (warmupSize < 1)   throw new IllegalStateException("warmupSize must be >= 1");
            return new StreamPipeline(this);
        }
    }
}