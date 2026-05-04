package thesis.evaluation;

import java.util.Locale;

public class MetricsCollector {

    private final int numClasses;
    private final int windowSize;
    private final int logEvery;
    private final int ramSampleEvery;

    private final CohenKappa kappa;
    private final TemporalKappa kappaPer;
    private final PrequentialAccuracy accuracy;
    private final RecoveryTime recovery;
    private final RAMHours ram;
    private final FeatureStabilityRatio stability;

    private long instances;
    private long totalUpdateNanos;
    private long correctTotal;
    private long driftCount;

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
        this.kappaPer = new TemporalKappa(windowSize);
        this.accuracy = new PrequentialAccuracy(windowSize);
        this.recovery = new RecoveryTime();
        this.ram = new RAMHours();
        this.stability = new FeatureStabilityRatio();
        this.ram.start();
    }

    public void update(int yTrue, int yPred, long elapsedNanos) {
        if (elapsedNanos < 0) elapsedNanos = 0;
        kappa.update(yTrue, yPred);
        kappaPer.update(yTrue, yPred);
        accuracy.update(yTrue, yPred);
        recovery.tick();
        recovery.update(kappa.getKappa());
        instances++;
        totalUpdateNanos += elapsedNanos;
        if (yTrue == yPred) correctTotal++;
        if (instances % ramSampleEvery == 0) ram.sampleFromRuntime();
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
                kappaPer.getKappaTemporal(),
                driftCount,
                recovery.getLastRecoveryTime(),
                recovery.getAverageRecoveryTime(),
                Double.isNaN(stab) ? 0.0 : stab,
                ram.getRamHours(),
                (long) ram.getPeakMB(),
                instances == 0 ? 0.0 : (totalUpdateNanos / 1000.0) / instances);
    }

    public Snapshot snapshot() {
        Snapshot s = new Snapshot();
        s.instances = instances;
        s.accuracyOverall = instances == 0 ? 0.0 : (double) correctTotal / instances;
        s.accuracyWindow = accuracy.getAccuracy();
        s.kappa = kappa.getKappa();
        s.kappaPer = kappaPer.getKappaTemporal();
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
        s.avgUpdateMicros = instances == 0 ? 0.0 : (totalUpdateNanos / 1000.0) / instances;
        return s;
    }

    public CohenKappa getKappa()                    { return kappa; }
    public TemporalKappa getKappaPer()              { return kappaPer; }
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
        public double kappaPer;
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
        public double avgUpdateMicros;
    }
}