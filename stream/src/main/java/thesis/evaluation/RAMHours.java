package thesis.evaluation;

public class RAMHours {

    private long startNanos;
    private long lastNanos;
    private long peakBytes;
    private double accumulatedRamHours;
    private long lastBytes;
    private boolean started;

    public void start() {
        startNanos = System.nanoTime();
        lastNanos = startNanos;
        peakBytes = 0;
        lastBytes = 0;
        accumulatedRamHours = 0.0;
        started = true;
    }

    public void sample(long usedBytes) {
        if (!started) start();
        long now = System.nanoTime();
        double hours = (now - lastNanos) / 3_600_000_000_000.0;
        double avgMB = ((lastBytes + usedBytes) / 2.0) / (1024.0 * 1024.0);
        accumulatedRamHours += avgMB * hours;
        lastNanos = now;
        lastBytes = usedBytes;
        if (usedBytes > peakBytes) peakBytes = usedBytes;
    }

    public void sampleFromRuntime() {
        Runtime r = Runtime.getRuntime();
        sample(r.totalMemory() - r.freeMemory());
    }

    public double getRamHours()     { return accumulatedRamHours; }
    public long getPeakBytes()      { return peakBytes; }
    public double getPeakMB()       { return peakBytes / (1024.0 * 1024.0); }
    public double getElapsedHours() { return started ? (System.nanoTime() - startNanos) / 3_600_000_000_000.0 : 0.0; }

    public double getPeakRamHours() { return getPeakMB() * getElapsedHours(); }

    public void reset() {
        started = false;
        startNanos = 0;
        lastNanos = 0;
        peakBytes = 0;
        lastBytes = 0;
        accumulatedRamHours = 0.0;
    }
}