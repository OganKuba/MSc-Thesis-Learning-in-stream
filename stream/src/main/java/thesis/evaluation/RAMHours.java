package thesis.evaluation;

public class RAMHours {

    private long startNanos;
    private long lastNanos;
    private long peakBytes;
    private double accumulatedGBHours;
    private long lastBytes;
    private boolean started;
    private boolean firstSample;

    public void start() {
        startNanos = System.nanoTime();
        lastNanos = startNanos;
        peakBytes = -1;
        lastBytes = 0;
        accumulatedGBHours = 0.0;
        started = true;
        firstSample = true;
    }

    public void sample(long usedBytes) {
        if (usedBytes < 0) throw new IllegalArgumentException("usedBytes < 0");
        if (!started) start();
        long now = System.nanoTime();
        if (firstSample) {
            firstSample = false;
        } else {
            long dt = now - lastNanos;
            if (dt > 0) {
                double hours = dt / 3_600_000_000_000.0;
                double avgGB = ((lastBytes + usedBytes) / 2.0) / (1024.0 * 1024.0 * 1024.0);
                accumulatedGBHours += avgGB * hours;
            }
        }
        lastNanos = now;
        lastBytes = usedBytes;
        if (peakBytes < 0 || usedBytes > peakBytes) peakBytes = usedBytes;
    }

    public void sampleFromRuntime() {
        Runtime r = Runtime.getRuntime();
        sample(r.totalMemory() - r.freeMemory());
    }

    public double getRamHours()     { return accumulatedGBHours; }
    public long getPeakBytes()      { return peakBytes < 0 ? 0 : peakBytes; }
    public double getPeakMB()       { return getPeakBytes() / (1024.0 * 1024.0); }
    public double getPeakGB()       { return getPeakBytes() / (1024.0 * 1024.0 * 1024.0); }
    public double getElapsedHours() { return started ? (System.nanoTime() - startNanos) / 3_600_000_000_000.0 : 0.0; }
    public double getPeakRamHours() { return getPeakGB() * getElapsedHours(); }

    public void reset() {
        started = false;
        firstSample = true;
        startNanos = 0;
        lastNanos = 0;
        peakBytes = -1;
        lastBytes = 0;
        accumulatedGBHours = 0.0;
    }
}