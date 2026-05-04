package thesis.pipeline;

import lombok.Getter;

@Getter
public class StreamMetrics {

    private long count;
    private long correct;
    private long totalTimeNanos;
    private long peakMemoryBytes;
    private long lastUpdateNanos;

    private long[][] confusion;
    private int numClasses;

    private long driftCount;
    private long lastDriftAt = -1;
    private long recoverySum;
    private long recoveryEvents;
    private boolean recovering;
    private long preDriftCorrect;
    private long preDriftCount;
    private double preDriftAcc;

    private int[] lastSelection;
    private long selectionChanges;
    private long selectionObservations;

    public StreamMetrics() {
        this(0);
    }

    public StreamMetrics(int numClasses) {
        if (numClasses > 0) {
            this.numClasses = numClasses;
            this.confusion = new long[numClasses][numClasses];
        }
    }

    public void update(int yTrue, int yHat, long elapsedNanos) {
        count++;
        if (yTrue == yHat) correct++;
        totalTimeNanos += elapsedNanos;
        lastUpdateNanos = elapsedNanos;
        if (confusion != null
                && yTrue >= 0 && yTrue < numClasses
                && yHat >= 0 && yHat < numClasses) {
            confusion[yTrue][yHat]++;
        }
        if (recovering) {
            long correctSince = correct - preDriftCorrect;
            long countSince = count - preDriftCount;
            if (countSince >= 50) {
                double accSince = (double) correctSince / countSince;
                if (accSince >= preDriftAcc - 0.01) {
                    recoverySum += (count - lastDriftAt);
                    recoveryEvents++;
                    recovering = false;
                }
            }
        }
    }

    public void onDriftAlarm() {
        driftCount++;
        lastDriftAt = count;
        preDriftCorrect = correct;
        preDriftCount = count;
        preDriftAcc = getAccuracy();
        recovering = true;
    }

    public void onSelectionChanged(int[] newSelection) {
        selectionObservations++;
        if (lastSelection != null && newSelection != null) {
            if (!java.util.Arrays.equals(lastSelection, newSelection)) {
                selectionChanges++;
            }
        }
        if (newSelection != null) lastSelection = newSelection.clone();
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
        if (confusion != null) {
            for (int i = 0; i < numClasses; i++)
                java.util.Arrays.fill(confusion[i], 0);
        }
        driftCount = 0;
        lastDriftAt = -1;
        recoverySum = 0;
        recoveryEvents = 0;
        recovering = false;
        preDriftAcc = 0.0;
        preDriftCorrect = 0;
        preDriftCount = 0;
        lastSelection = null;
        selectionChanges = 0;
        selectionObservations = 0;
    }

    public double getAccuracy() {
        return count == 0 ? 0.0 : (double) correct / count;
    }

    public double getAvgTimeMicros() {
        return count == 0 ? 0.0 : totalTimeNanos / 1000.0 / count;
    }

    public double getKappa() {
        if (confusion == null || count == 0) return 0.0;
        double po = getAccuracy();
        double pe = 0.0;
        for (int i = 0; i < numClasses; i++) {
            long rowSum = 0, colSum = 0;
            for (int j = 0; j < numClasses; j++) {
                rowSum += confusion[i][j];
                colSum += confusion[j][i];
            }
            pe += ((double) rowSum / count) * ((double) colSum / count);
        }
        if (pe >= 1.0) return 0.0;
        return (po - pe) / (1.0 - pe);
    }

    public double getKappaPer() {
        if (confusion == null || count < 2) return 0.0;
        double po = getAccuracy();
        long agree = 0;
        long prev = -1;
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                long c = confusion[i][j];
                for (long k = 0; k < c; k++) {
                    long curr = i;
                    if (prev == curr) agree++;
                    prev = curr;
                }
            }
        }
        double pePer = (double) agree / (count - 1);
        if (pePer >= 1.0) return 0.0;
        return (po - pePer) / (1.0 - pePer);
    }

    public double getRamHours() {
        double gbHours = (peakMemoryBytes / (1024.0 * 1024.0 * 1024.0))
                * (totalTimeNanos / 3.6e12);
        return gbHours;
    }

    public double getAvgRecoveryTime() {
        return recoveryEvents == 0 ? Double.NaN : (double) recoverySum / recoveryEvents;
    }

    public double getFeatureStabilityRatio() {
        if (selectionObservations <= 1) return 1.0;
        return 1.0 - ((double) selectionChanges / (selectionObservations - 1));
    }

    public long getPeakMB() {
        return peakMemoryBytes / (1024 * 1024);
    }
}