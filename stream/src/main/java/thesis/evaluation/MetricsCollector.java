package thesis.evaluation;

public class MetricsCollector {

    private final int numClasses;
    private final int windowSize;
    private final int logEvery;

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

    public MetricsCollector(int numClasses) { this(numClasses, 1000, 1000); }

    public MetricsCollector(int numClasses, int windowSize, int logEvery) {
        this.numClasses = numClasses;
        this.windowSize = windowSize;
        this.logEvery = logEvery;
        this.kappa = new CohenKappa(numClasses, windowSize);
        this.kappaPer = new TemporalKappa(windowSize);
        this.accuracy = new PrequentialAccuracy(windowSize);
        this.recovery = new RecoveryTime();
        this.ram = new RAMHours();
        this.stability = new FeatureStabilityRatio();
        this.ram.start();
    }

    public void update(int yTrue, int yPred, long elapsedNanos) {
        kappa.update(yTrue, yPred);
        kappaPer.update(yTrue, yPred);
        accuracy.update(yTrue, yPred);
        recovery.tick();
        recovery.update(kappa.getKappa());
        instances++;
        totalUpdateNanos += elapsedNanos;
        if (yTrue == yPred) correctTotal++;
        if (instances % 100 == 0) ram.sampleFromRuntime();
    }

    public void onDriftAlarm() {
        driftCount++;
        recovery.onDriftAlarm(kappa.getKappa());
    }

    public void onSelectionChanged(int[] currentSelection) {
        stability.update(currentSelection);
    }

    public boolean shouldLog() {
        return logEvery > 0 && instances > 0 && instances % logEvery == 0;
    }

    public String formatLogLine() {
        return String.format(
                "[t=%6d] acc=%.4f  κ=%.4f  κ_per=%.4f  drift=%d  recov(last/avg)=%d/%.1f  stab=%.3f  ramH=%.4f  peak=%dMB  avg=%.1fµs",
                instances,
                accuracy.getAccuracy(),
                kappa.getKappa(),
                kappaPer.getKappaTemporal(),
                driftCount,
                recovery.getLastRecoveryTime(),
                recovery.getAverageRecoveryTime(),
                Double.isNaN(stability.getAverageRatio()) ? 0.0 : stability.getAverageRatio(),
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
        s.featureStabilityRatio = stability.getAverageRatio();
        s.ramHours = ram.getRamHours();
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
        public double featureStabilityRatio;
        public double ramHours;
        public double peakMB;
        public double elapsedHours;
        public double avgUpdateMicros;
    }
}