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

    private boolean warmedUp;
    private long processed;
    private long globalAlarmsSeen;

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

        if (this.source instanceof moa.options.OptionHandler) {
            ((moa.options.OptionHandler) this.source).prepareForUse();
        }
        InstancesHeader header = source.getHeader();
        if (header == null) {
            throw new IllegalStateException(
                    "source stream has no header — make sure prepareForUse() was called on the concrete stream before passing it to the pipeline");
        }
        this.space = new FeatureSpace(header);
        this.warmedUp = selector.isInitialized();
    }

    public static Builder builder() { return new Builder(); }

    public void run() {
        warmupIfNeeded();
        while (source.hasMoreInstances() && processed < maxInstances) {
            Instance x = source.nextInstance().getData();
            processInstance(x);
        }
        if (verbose) finalLog();
    }

    public void processInstance(Instance raw) {
        if (!warmedUp) {
            throw new IllegalStateException("pipeline not warmed up — call warmupIfNeeded() or run()");
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
        Set<Integer> drifting = alarm ? detector.getDriftingFeatureIndices() : Set.of();

        if (alarm) {
            globalAlarmsSeen++;
            if (rankingPid != null) {
                for (int idx : drifting) rankingPid.resetFeature(idx);
            }
            if (importance != null && fullRanker != null) {
                importance.update(fullRanker.getFeatureScores(), invertPValues(detector.getLastPValues()));
            }
        }

        model.train(raw, y, alarm, drifting);

        if (alarm && model instanceof DriftAwareSRP) {
            double[] scores = (fullRanker != null)
                    ? fullRanker.getFeatureScores()
                    : new double[space.numFeatures()];
            ((DriftAwareSRP) model).handleDrift(drifting, scores);
        }
        if (refreshEvery > 0 && processed > 0 && processed % refreshEvery == 0
                && model instanceof DriftAwareSRP) {
            ((DriftAwareSRP) model).refreshAllSubspaces();
        }

        long elapsed = System.nanoTime() - t0;
        metrics.update(y, yhat, elapsed);
        processed++;

        if (verbose && logEvery > 0 && processed % logEvery == 0) {
            log();
        }
    }

    public void warmupIfNeeded() {
        if (warmedUp) return;
        if (selector.isInitialized()) {
            warmedUp = true;
            return;
        }

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
        if (collected == 0) {
            throw new IllegalStateException("source stream produced no instances during warmup");
        }
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
            importance.update(fullRanker.getFeatureScores(),
                    new double[space.numFeatures()]);
        }

        warmedUp = true;
        if (verbose) {
            System.out.printf("[warmup] collected=%d  selector=%s  initial selection=%s%n",
                    collected, selector.name(),
                    Arrays.toString(selector.getCurrentSelection()));
        }
    }

    private void log() {
        Runtime r = Runtime.getRuntime();
        long used = r.totalMemory() - r.freeMemory();
        metrics.recordMemory(used);
        System.out.printf("[t=%6d] acc=%.4f  avg=%.1fµs  alarms=%d  selection=%s  mem=%dMB%n",
                processed, metrics.getAccuracy(), metrics.getAvgTimeMicros(),
                globalAlarmsSeen, Arrays.toString(selector.getCurrentSelection()),
                used / (1024 * 1024));
    }

    private void finalLog() {
        Runtime r = Runtime.getRuntime();
        long used = r.totalMemory() - r.freeMemory();
        metrics.recordMemory(used);
        System.out.println("─".repeat(80));
        System.out.println("FINAL");
        System.out.printf("  processed=%d  acc=%.4f  avgTime=%.2fµs  globalAlarms=%d%n",
                processed, metrics.getAccuracy(), metrics.getAvgTimeMicros(), globalAlarmsSeen);
        System.out.printf("  peakMem=%dMB  finalSelection=%s%n",
                metrics.getPeakMemoryBytes() / (1024 * 1024),
                Arrays.toString(selector.getCurrentSelection()));
        System.out.println("  model: " + model.name());
        if (model instanceof DriftAwareSRP) {
            DriftAwareSRP da = (DriftAwareSRP) model;
            System.out.printf("  DA-SRP: handleDriftCalls=%d  keep=%d  surgical=%d  full=%d  noRepl=%d  refresh=%d/%d  weighted/fallback=%d/%d%n",
                    da.getHandleDriftCalls(), da.getTotalKept(), da.getTotalSurgical(),
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

        public Builder source(InstanceStream s)              { this.source = s; return this; }
        public Builder selector(FeatureSelector s)           { this.selector = s; return this; }
        public Builder model(ModelWrapper m)                 { this.model = m; return this; }
        public Builder detector(TwoLevelDriftDetector d)     { this.detector = d; return this; }
        public Builder metrics(StreamMetrics m)              { this.metrics = m; return this; }
        public Builder rankingPid(PiDDiscretizer p)          { this.rankingPid = p; return this; }
        public Builder fullRanker(FilterRanker r)            { this.fullRanker = r; return this; }
        public Builder importance(FeatureImportance i)       { this.importance = i; return this; }
        public Builder warmup(int n)                         { this.warmupSize = n; return this; }
        public Builder logEvery(int n)                       { this.logEvery = n; return this; }
        public Builder refreshEvery(int n)                   { this.refreshEvery = n; return this; }
        public Builder maxInstances(long n)                  { this.maxInstances = n; return this; }
        public Builder verbose(boolean v)                    { this.verbose = v; return this; }

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