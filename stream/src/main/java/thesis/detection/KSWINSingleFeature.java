package thesis.detection;

import lombok.Getter;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

@Getter
public class KSWINSingleFeature {

    private final int windowSize;
    private final double alpha;
    private final KolmogorovSmirnovTest ks;

    private double[] reference;
    private final Deque<Double> current;

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
        this.current = new ArrayDeque<>(windowSize);
        this.lastPValue = 1.0;
        this.lastKsStatistic = 0.0;
        this.lastDrift = false;
    }

    public void update(double value) {
        if (reference == null) {
            current.addLast(value);
            if (current.size() == windowSize) {
                reference = drainToArray();
            }
            return;
        }
        if (current.size() == windowSize) {
            current.removeFirst();
        }
        current.addLast(value);
    }

    public boolean testDrift() {
        if (reference == null || current.size() < windowSize) {
            lastPValue = 1.0;
            lastKsStatistic = 0.0;
            lastDrift = false;
            return false;
        }
        double[] cur = toArray(current);
        lastKsStatistic = ks.kolmogorovSmirnovStatistic(reference, cur);
        lastPValue = ks.kolmogorovSmirnovTest(reference, cur);
        lastDrift = lastPValue < alpha;
        return lastDrift;
    }

    public double getPValue()      { return lastPValue; }
    public double getKSStatistic() { return lastKsStatistic; }
    public boolean isDrift()       { return lastDrift; }
    public boolean isReady()       { return reference != null && current.size() == windowSize; }

    public void setReferenceWindow(double[] ref) {
        if (ref == null || ref.length != windowSize) {
            throw new IllegalArgumentException(
                    "reference must have length windowSize=" + windowSize);
        }
        this.reference = Arrays.copyOf(ref, ref.length);
        this.current.clear();
        this.lastPValue = 1.0;
        this.lastKsStatistic = 0.0;
        this.lastDrift = false;
    }

    public void promoteCurrentToReference() {
        if (current.size() != windowSize) {
            throw new IllegalStateException("current window not full yet");
        }
        this.reference = toArray(current);
        this.current.clear();
        this.lastPValue = 1.0;
        this.lastKsStatistic = 0.0;
        this.lastDrift = false;
    }

    public void reset() {
        this.reference = null;
        this.current.clear();
        this.lastPValue = 1.0;
        this.lastKsStatistic = 0.0;
        this.lastDrift = false;
    }

    private double[] drainToArray() {
        double[] out = toArray(current);
        current.clear();
        return out;
    }

    private static double[] toArray(Deque<Double> dq) {
        double[] out = new double[dq.size()];
        int i = 0;
        for (Double v : dq) out[i++] = v;
        return out;
    }
}