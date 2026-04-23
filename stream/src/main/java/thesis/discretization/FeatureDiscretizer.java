package thesis.discretization;

import lombok.Getter;

@Getter
public class FeatureDiscretizer {

    private final int b1;
    private final int b2;
    private final int numClasses;
    private final int warmupN;

    private boolean ready;
    private double[] warmupBuffer;
    private int[] warmupClasses;
    private int warmupCount;

    private Layer1Histogram l1;
    private int[] l1ToL2;
    private long updatesSinceRecompute;
    private long totalUpdates;

    public FeatureDiscretizer(int b1, int b2, int numClasses, int warmupN) {
        if (b1 < 4) throw new IllegalArgumentException("b1 must be >= 4");
        if (b2 < 2 || b2 > b1) throw new IllegalArgumentException("require 2 <= b2 <= b1");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        if (warmupN < b1) throw new IllegalArgumentException("warmupN must be >= b1");
        this.b1 = b1;
        this.b2 = b2;
        this.numClasses = numClasses;
        this.warmupN = warmupN;
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
    }

    public void update(double value, int classLabel) {
        totalUpdates++;
        if (!ready) {
            warmupBuffer[warmupCount] = value;
            warmupClasses[warmupCount] = classLabel;
            warmupCount++;
            if (warmupCount == warmupN) initializeFromWarmup();
            return;
        }
        l1.update(value, classLabel);
        updatesSinceRecompute++;
    }

    private void initializeFromWarmup() {
        l1 = Layer1Histogram.fromWarmup(warmupBuffer, warmupClasses, b1, numClasses);
        l1ToL2 = uniformMapping(b1, b2);
        ready = true;
        warmupBuffer = null;
        warmupClasses = null;
        updatesSinceRecompute = warmupN;
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
        if (!ready) return 0;
        return l1ToL2[l1.bin(value)];
    }

    public void recomputeLayer2() {
        if (!ready) return;
        l1ToL2 = Layer2Merger.merge(l1.getBinCounts(), l1.getClassCounts(), b2, numClasses);
        updatesSinceRecompute = 0;
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

    public Layer1Histogram getLayer1()         { return l1; }
    public int[] getL1ToL2Mapping()            { return l1ToL2; }
}