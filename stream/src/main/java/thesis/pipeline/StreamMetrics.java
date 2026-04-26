package thesis.pipeline;

import lombok.Getter;

@Getter
public class StreamMetrics {

    private long count;
    private long correct;
    private long totalTimeNanos;
    private long peakMemoryBytes;
    private long lastUpdateNanos;

    public void update(int yTrue, int yHat, long elapsedNanos) {
        count++;
        if (yTrue == yHat) correct++;
        totalTimeNanos += elapsedNanos;
        lastUpdateNanos = elapsedNanos;
    }

    public void recordMemory(long bytes) {
        if (bytes > peakMemoryBytes) peakMemoryBytes = bytes;
    }

    public void reset() {
        count = 0;
        correct = 0;
        totalTimeNanos = 0;
        peakMemoryBytes = 0;
        lastUpdateNanos = 0;
    }

    public double getAccuracy()          { return count == 0 ? 0.0 : (double) correct / count; }
    public double getAvgTimeMicros()     { return count == 0 ? 0.0 : totalTimeNanos / 1000.0 / count; }
}