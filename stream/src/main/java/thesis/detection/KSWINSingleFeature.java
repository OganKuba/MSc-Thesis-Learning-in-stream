package thesis.detection;

import lombok.Getter;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;

import java.util.Arrays;

@Getter
public class KSWINSingleFeature {

    private final int windowSize;
    private final double alpha;
    private final KolmogorovSmirnovTest ks;

    private double[] reference;
    private final double[] current;
    private int currentSize;
    private int currentHead;
    private final double[] currentScratch;

    private double lastPValue;
    private double lastKsStatistic;
    private boolean lastDrift;

    public KSWINSingleFeature(int windowSize, double alpha) {
        if (windowSize < 10) {
            throw new IllegalArgumentException("windowSize must be >= 10, got " + windowSize);
        }
        if (!(alpha > 0.0 && alpha < 1.0)) {
            throw new IllegalArgumentException("alpha must be in (0,1), got " + alpha);
        }
        this.windowSize = windowSize;
        this.alpha = alpha;
        this.ks = new KolmogorovSmirnovTest();
        this.reference = null;
        this.current = new double[windowSize];
        this.currentScratch = new double[windowSize];
        this.currentSize = 0;
        this.currentHead = 0;
        this.lastPValue = 1.0;
        this.lastKsStatistic = 0.0;
        this.lastDrift = false;
    }

    public void update(double value) {
        if (!Double.isFinite(value)) return;
        if (reference == null) {
            current[currentSize++] = value;
            if (currentSize == windowSize) {
                reference = Arrays.copyOf(current, windowSize);
                currentSize = 0;
                currentHead = 0;
            }
            return;
        }
        if (currentSize < windowSize) {
            current[(currentHead + currentSize) % windowSize] = value;
            currentSize++;
        } else {
            current[currentHead] = value;
            currentHead = (currentHead + 1) % windowSize;
        }
    }

    public boolean testDrift() {
        if (reference == null || currentSize < windowSize) {
            lastPValue = 1.0;
            lastKsStatistic = 0.0;
            lastDrift = false;
            return false;
        }
        for (int i = 0; i < windowSize; i++) {
            currentScratch[i] = current[(currentHead + i) % windowSize];
        }
        lastKsStatistic = ks.kolmogorovSmirnovStatistic(reference, currentScratch);
        lastPValue = ks.kolmogorovSmirnovTest(reference, currentScratch);
        lastDrift = lastPValue < alpha;
        return lastDrift;
    }

    public double getPValue()      { return lastPValue; }
    public double getKSStatistic() { return lastKsStatistic; }
    public boolean isDrift()       { return lastDrift; }
    public boolean isReady()       { return reference != null && currentSize == windowSize; }

    public void setReferenceWindow(double[] ref) {
        if (ref == null || ref.length != windowSize) {
            throw new IllegalArgumentException(
                    "reference must have length windowSize=" + windowSize);
        }
        this.reference = Arrays.copyOf(ref, ref.length);
        this.currentSize = 0;
        this.currentHead = 0;
        this.lastPValue = 1.0;
        this.lastKsStatistic = 0.0;
        this.lastDrift = false;
    }

    public void promoteCurrentToReference() {
        if (currentSize != windowSize) {
            throw new IllegalStateException("current window not full yet");
        }
        double[] newRef = new double[windowSize];
        for (int i = 0; i < windowSize; i++) {
            newRef[i] = current[(currentHead + i) % windowSize];
        }
        this.reference = newRef;
        this.currentSize = 0;
        this.currentHead = 0;
        this.lastPValue = 1.0;
        this.lastKsStatistic = 0.0;
        this.lastDrift = false;
    }

    public void reset() {
        this.reference = null;
        this.currentSize = 0;
        this.currentHead = 0;
        this.lastPValue = 1.0;
        this.lastKsStatistic = 0.0;
        this.lastDrift = false;
    }
}