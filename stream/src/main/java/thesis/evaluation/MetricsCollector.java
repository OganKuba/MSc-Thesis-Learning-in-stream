package thesis.evaluation;

import java.util.Locale;

public class MetricsCollector {

    private final int numClasses;
    private final int windowSize;
    private final int logEvery;
    private final int ramSampleEvery;

    private final CohenKappa kappa;
    private final TemporalKappa kappaTemporal;
    private final PrequentialAccuracy accuracy;
    private final RecoveryTime recovery;
    private final RAMHours ram;
    private final FeatureStabilityRatio stability;

    private long instances;
    /** Time spent inside {@code model.predict()} only — inference latency. */
    private long totalPredictNanos;
    /** Time for the whole prequential step: predict + drift detection + selection + training. */
    private long totalStepNanos;
    private long correctTotal;
    private long driftCount;

    /**
     * Supplies the model's deep byte size for RAM-Hours. When left unset the collector falls back
     * to the legacy whole-JVM reading, which is only meaningful for single-threaded diagnostics —
     * the experiment runner always installs a supplier.
     */
    private java.util.function.LongSupplier modelSizeSupplier;

    public MetricsCollector(int numClasses) { this(numClasses, 1000, 1000, 100); }

    public MetricsCollector(int numClasses, int windowSize, int logEvery, int ramSampleEvery) {
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        if (windowSize < 1) throw new IllegalArgumentException("windowSize must be >= 1");
        if (ramSampleEvery < 1) throw new IllegalArgumentException("ramSampleEvery must be >= 1");
        this.numClasses = numClasses;
        this.windowSize = windowSize;
        this.logEvery = logEvery;
        this.ramSampleEvery = ramSampleEvery;
        this.kappa = new CohenKappa(numClasses, windowSize);
        this.kappaTemporal = new TemporalKappa(windowSize);
        this.accuracy = new PrequentialAccuracy(windowSize);
        this.recovery = new RecoveryTime();
        this.ram = new RAMHours();
        this.stability = new FeatureStabilityRatio();
        this.ram.start();
    }

    /**
     * Legacy two-timer-in-one form: the supplied duration is charged as both the inference
     * latency and the cost of the whole step. Only appropriate where nothing else happens per
     * instance; the experiment runner uses the four-argument form.
     */
    public void update(int yTrue, int yPred, long elapsedNanos) {
        update(yTrue, yPred, elapsedNanos, elapsedNanos);
    }

    /**
     * @param predictNanos time inside {@code model.predict()} — inference latency
     * @param stepNanos    time for the entire prequential step (predict + detector + selector +
     *                     train). Throughput must be derived from this one: deriving it from
     *                     {@code predictNanos} answers "how fast could this model predict if it
     *                     never learned anything", which for a majority-class baseline came out at
     *                     ~40 million instances/s.
     */
    public void update(int yTrue, int yPred, long predictNanos, long stepNanos) {
        if (predictNanos < 0) predictNanos = 0;
        if (stepNanos < 0) stepNanos = 0;
        kappa.update(yTrue, yPred);
        kappaTemporal.update(yTrue, yPred);
        accuracy.update(yTrue, yPred);
        recovery.tick();
        recovery.update(kappa.getKappa());
        instances++;
        totalPredictNanos += predictNanos;
        totalStepNanos += stepNanos;
        if (yTrue == yPred) correctTotal++;
        if (instances % ramSampleEvery == 0) sampleRam();
    }

    @SuppressWarnings("deprecation")
    private void sampleRam() {
        if (modelSizeSupplier != null) ram.sampleModelSize(modelSizeSupplier.getAsLong());
        else ram.sampleFromRuntime();
    }

    /**
     * Install the model-size source for RAM-Hours (typically {@code model::modelByteSize}).
     * Without it the collector measures the whole JVM heap, which is shared across concurrently
     * running experiments and therefore not attributable to any one model.
     */
    public void setModelSizeSupplier(java.util.function.LongSupplier supplier) {
        this.modelSizeSupplier = supplier;
    }

    public void onDriftAlarm() {
        driftCount++;
        recovery.onDriftAlarm(kappa.getKappa());
    }

    public boolean onSelectionChanged(int[] currentSelection) {
        stability.update(currentSelection);
        return stability.wasLastChanged();
    }

    public boolean shouldLog() {
        return logEvery > 0 && instances > 0 && instances % logEvery == 0;
    }

    public String formatLogLine() {
        double stab = stability.getAverageRatio();
        return String.format(Locale.ROOT,
                "[t=%6d] acc=%.4f  k=%.4f  k_per=%.4f  drift=%d  recov(last/avg)=%d/%.1f  stab=%.3f  ramH(GB)=%.6f  peak=%dMB  avg=%.1fus",
                instances,
                accuracy.getAccuracy(),
                kappa.getKappa(),
                kappaTemporal.getKappaTemporal(),
                driftCount,
                recovery.getLastRecoveryTime(),
                recovery.getAverageRecoveryTime(),
                Double.isNaN(stab) ? 0.0 : stab,
                ram.getRamHours(),
                (long) ram.getPeakMB(),
                instances == 0 ? 0.0 : (totalStepNanos / 1000.0) / instances);
    }

    public Snapshot snapshot() {
        Snapshot s = new Snapshot();
        s.instances = instances;
        s.accuracyOverall = instances == 0 ? 0.0 : (double) correctTotal / instances;
        s.accuracyWindow = accuracy.getAccuracy();
        s.kappa = kappa.getKappa();
        s.kappaTemporal = kappaTemporal.getKappaTemporal();
        s.driftCount = driftCount;
        s.lastRecoveryTime = recovery.getLastRecoveryTime();
        s.avgRecoveryTime = recovery.getAverageRecoveryTime();
        s.recovered = recovery.getRecoveredCount();
        s.unrecovered = recovery.getUnrecoveredCount();
        s.cancelled = recovery.getCancelledCount();
        s.featureStabilityRatio = stability.getAverageRatio();
        s.lastFeatureStabilityRatio = stability.getLastRatio();
        s.selectionChangeCount = stability.getChangeCount();
        s.ramHoursGB = ram.getRamHours();
        s.peakMB = ram.getPeakMB();
        s.elapsedHours = ram.getElapsedHours();
        s.avgPredictMicros = instances == 0 ? 0.0 : (totalPredictNanos / 1000.0) / instances;
        s.avgStepMicros    = instances == 0 ? 0.0 : (totalStepNanos / 1000.0) / instances;
        return s;
    }

    public CohenKappa getKappa()                    { return kappa; }
    public TemporalKappa getKappaTemporal()         { return kappaTemporal; }
    public PrequentialAccuracy getAccuracy()        { return accuracy; }
    public RecoveryTime getRecovery()               { return recovery; }
    public RAMHours getRam()                        { return ram; }
    public FeatureStabilityRatio getStability()     { return stability; }
    public long getInstances()                      { return instances; }
    public long getDriftCount()                     { return driftCount; }
    public int getNumClasses()                      { return numClasses; }
    public int getWindowSize()                      { return windowSize; }

    public static final class Snapshot {
        public long instances;
        public double accuracyOverall;
        public double accuracyWindow;
        public double kappa;
        /** Temporal kappa over the most recent window (a.k.a. kappa_per / kappa_+). */
        public double kappaTemporal;
        public long driftCount;
        public int lastRecoveryTime;
        public double avgRecoveryTime;
        public int recovered;
        public int unrecovered;
        public int cancelled;
        public double featureStabilityRatio;
        public double lastFeatureStabilityRatio;
        public long selectionChangeCount;
        public double ramHoursGB;
        public double peakMB;
        public double elapsedHours;
        /** Mean {@code model.predict()} latency in microseconds. */
        public double avgPredictMicros;
        /** Mean cost of a full prequential step in microseconds — the basis for throughput. */
        public double avgStepMicros;
    }
}