package thesis.discretization;

import lombok.Getter;

@Getter
public class FeatureDiscretizer {

    public static final int UNKNOWN_BIN = -1;

    private final int b1;
    private final int b2;
    private final int numClasses;
    private final int warmupN;
    private final double expandThreshold;
    private final double decayFactor;

    private boolean ready;
    private double[] warmupBuffer;
    private int[] warmupClasses;
    private int warmupCount;

    private Layer1Histogram l1;
    private int[] l1ToL2;
    private long updatesSinceRecompute;
    private long totalUpdates;
    private long expansions;

    public FeatureDiscretizer(int b1, int b2, int numClasses, int warmupN) {
        this(b1, b2, numClasses, warmupN, 0.20, 1.0);
    }

    public FeatureDiscretizer(int b1, int b2, int numClasses, int warmupN,
                              double expandThreshold, double decayFactor) {
        if (b1 < 4) throw new IllegalArgumentException("b1 must be >= 4");
        if (b2 < 2 || b2 > b1) throw new IllegalArgumentException("require 2 <= b2 <= b1");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        if (warmupN < b1) throw new IllegalArgumentException("warmupN must be >= b1");
        if (!(expandThreshold > 0.0 && expandThreshold < 1.0))
            throw new IllegalArgumentException("expandThreshold must be in (0,1)");
        if (!(decayFactor > 0.0 && decayFactor <= 1.0))
            throw new IllegalArgumentException("decayFactor must be in (0,1]");
        this.b1 = b1;
        this.b2 = b2;
        this.numClasses = numClasses;
        this.warmupN = warmupN;
        this.expandThreshold = expandThreshold;
        this.decayFactor = decayFactor;
        reset();
    }

    public void reset() {
        ready = false;
        warmupBuffer = new double[warmupN];
        warmupClasses = new int[warmupN];
        warmupCount = 0;
        l1 = null;
        l1ToL2 = null;
        updatesSinceRecompute = 0;
        totalUpdates = 0;
        expansions = 0;
    }

    public void update(double value, int classLabel) {
        if (classLabel < 0 || classLabel >= numClasses) {
            throw new IllegalArgumentException("classLabel out of range: " + classLabel);
        }
        if (!Double.isFinite(value)) return;
        totalUpdates++;
        if (!ready) {
            warmupBuffer[warmupCount] = value;
            warmupClasses[warmupCount] = classLabel;
            warmupCount++;
            if (warmupCount == warmupN) initializeFromWarmup();
            return;
        }
        if (value < l1.getMin() || value >= l1.getMax()) {
            expandRange(value);
        }
        l1.update(value, classLabel);
        updatesSinceRecompute++;
        if (l1.shouldExpand(expandThreshold)) {
            expandRange(value);
        }
    }

    private void expandRange(double value) {
        double mn = Math.min(l1.getMin(), value);
        double mx = Math.max(l1.getMax(), value);
        double pad = Math.max((mx - mn) * 0.20, 1e-9);
        l1.rebin(mn - pad, mx + pad);
        expansions++;
    }

    private void initializeFromWarmup() {
        l1 = Layer1Histogram.fromWarmup(warmupBuffer, warmupClasses, b1, numClasses);
        l1ToL2 = uniformMapping(b1, b2);
        ready = true;
        warmupBuffer = null;
        warmupClasses = null;
        updatesSinceRecompute = 0;
        recomputeLayer2();
    }

    private static int[] uniformMapping(int b1, int b2) {
        int[] m = new int[b1];
        for (int i = 0; i < b1; i++) {
            int g = (int) Math.floor(((double) i * b2) / b1);
            if (g >= b2) g = b2 - 1;
            m[i] = g;
        }
        return m;
    }

    public int discretize(double value) {
        if (!ready) return UNKNOWN_BIN;
        if (!Double.isFinite(value)) return UNKNOWN_BIN;
        return l1ToL2[l1.bin(value)];
    }

    public void recomputeLayer2() {
        if (!ready) return;
        if (decayFactor < 1.0) l1.decay(decayFactor);
        l1ToL2 = Layer2Merger.merge(l1.getBinCounts(), l1.getClassCounts(), b2, numClasses);
        updatesSinceRecompute = 0;
    }

    public void softReset(double decay) {
        if (!ready) return;
        if (!(decay > 0.0 && decay < 1.0)) throw new IllegalArgumentException("decay in (0,1)");
        l1.decay(decay);
        recomputeLayer2();
    }

    public int[] l2Counts() {
        int[] out = new int[b2];
        if (!ready) return out;
        int[] cnt = l1.getBinCounts();
        for (int i = 0; i < b1; i++) out[l1ToL2[i]] += cnt[i];
        return out;
    }

    public int[][] l2ClassCounts() {
        int[][] out = new int[b2][numClasses];
        if (!ready) return out;
        int[][] cc = l1.getClassCounts();
        for (int i = 0; i < b1; i++) {
            for (int c = 0; c < numClasses; c++) {
                out[l1ToL2[i]][c] += cc[i][c];
            }
        }
        return out;
    }

    public Layer1Histogram getLayer1() { return l1; }
    public int[] getL1ToL2Mapping()    { return l1ToL2; }
}