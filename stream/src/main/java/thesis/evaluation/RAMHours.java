package thesis.evaluation;

public class RAMHours {

    private long startNanos;
    private long lastNanos;
    private long peakBytes;
    private double accumulatedGBHours;
    private long lastBytes;
    private boolean started;
    private boolean firstSample;
    private long negativeSampleCount;
    private boolean unavailable;

    public void start() {
        startNanos = System.nanoTime();
        lastNanos = startNanos;
        peakBytes = -1;
        lastBytes = 0;
        accumulatedGBHours = 0.0;
        started = true;
        firstSample = true;
        negativeSampleCount = 0;
        unavailable = false;
    }

    public void sampleModelSize(long modelBytes) {
        if (!started) start();
        if (modelBytes < 0) {
            unavailable = true;
            return;
        }
        sample(modelBytes);
    }

    public boolean isUnavailable() { return unavailable; }

    public void sample(long usedBytes) {
        if (!started) start();
        if (usedBytes < 0) {
            if (negativeSampleCount == 0) {
                System.err.printf(
                        "[RAMHours][WARN] negative usedBytes=%d clamped to 0 "
                                + "(transient GC / non-atomic Runtime read); further occurrences suppressed%n",
                        usedBytes);
            }
            negativeSampleCount++;
            usedBytes = 0L;
        }
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

    @Deprecated
    public void sampleFromRuntime() {
        Runtime r = Runtime.getRuntime();
        long total = r.totalMemory();
        long free  = r.freeMemory();
        long used  = total - free;
        sample(used < 0 ? 0L : used);
    }

    public long getNegativeSampleCount() { return negativeSampleCount; }

    public double getRamHours()     { return unavailable ? Double.NaN : accumulatedGBHours; }
    public long getPeakBytes()      { return peakBytes < 0 ? 0 : peakBytes; }
    public double getPeakMB()       { return unavailable ? Double.NaN : getPeakBytes() / (1024.0 * 1024.0); }
    public double getPeakGB()       { return unavailable ? Double.NaN : getPeakBytes() / (1024.0 * 1024.0 * 1024.0); }
    public double getElapsedHours() { return started ? (System.nanoTime() - startNanos) / 3_600_000_000_000.0 : 0.0; }
    public double getPeakRamHours() { return getPeakGB() * getElapsedHours(); }

    public void reset() {
        started = false;
        firstSample = true;
        unavailable = false;
        startNanos = 0;
        lastNanos = 0;
        peakBytes = -1;
        lastBytes = 0;
        accumulatedGBHours = 0.0;
        negativeSampleCount = 0;
    }
}