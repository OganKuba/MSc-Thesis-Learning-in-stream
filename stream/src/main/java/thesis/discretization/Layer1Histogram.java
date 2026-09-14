package thesis.discretization;

import lombok.Getter;

@Getter
public class Layer1Histogram {

    private final int b1;
    private final int numClasses;

    private double min;
    private double max;
    private double width;
    private int[] binCounts;
    private int[][] classCounts;

    private long underflow;
    private long overflow;

    public Layer1Histogram(int b1, int numClasses, double min, double max) {
        if (b1 < 4) throw new IllegalArgumentException("b1 must be >= 4");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        if (!(max > min)) throw new IllegalArgumentException("max must be > min");
        this.b1 = b1;
        this.numClasses = numClasses;
        this.min = min;
        this.max = max;
        this.width = (max - min) / b1;
        this.binCounts = new int[b1];
        this.classCounts = new int[b1][numClasses];
    }

    public static Layer1Histogram fromWarmup(double[] values, int[] classes,
                                             int b1, int numClasses) {
        double mn = Double.POSITIVE_INFINITY;
        double mx = Double.NEGATIVE_INFINITY;
        for (double v : values) {
            if (!Double.isFinite(v)) continue;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        if (!(mn < mx)) { mn -= 0.5; mx += 0.5; }
        double margin = Math.max((mx - mn) * 0.10, 1e-9);
        Layer1Histogram h = new Layer1Histogram(b1, numClasses, mn - margin, mx + margin);
        for (int i = 0; i < values.length; i++) {
            if (Double.isFinite(values[i])) h.update(values[i], classes[i]);
        }
        return h;
    }

    public void update(double value, int classLabel) {
        if (value < min) { underflow++; binCounts[0]++; classCounts[0][classLabel]++; return; }
        if (value >= max) { overflow++; int last = b1 - 1; binCounts[last]++; classCounts[last][classLabel]++; return; }
        int idx = (int) Math.floor((value - min) / width);
        if (idx < 0) idx = 0;
        if (idx >= b1) idx = b1 - 1;
        binCounts[idx]++;
        classCounts[idx][classLabel]++;
    }

    public int bin(double value) {
        if (width <= 0.0) return 0;
        int idx = (int) Math.floor((value - min) / width);
        if (idx < 0) idx = 0;
        if (idx >= b1) idx = b1 - 1;
        return idx;
    }

    public boolean shouldExpand(double saturationThreshold) {
        long total = totalCount();
        if (total < 50) return false;
        long edges = (long) binCounts[0] + binCounts[b1 - 1];
        return (double) edges / total > saturationThreshold;
    }

    public long totalCount() {
        long s = 0;
        for (int c : binCounts) s += c;
        return s;
    }

    public void rebin(double newMin, double newMax) {
        if (!(newMax > newMin)) throw new IllegalArgumentException("newMax must be > newMin");
        double newWidth = (newMax - newMin) / b1;
        int[] nb = new int[b1];
        int[][] nc = new int[b1][numClasses];
        for (int i = 0; i < b1; i++) {
            if (binCounts[i] == 0) continue;
            double center = min + (i + 0.5) * width;
            int j = (int) Math.floor((center - newMin) / newWidth);
            if (j < 0) j = 0;
            if (j >= b1) j = b1 - 1;
            nb[j] += binCounts[i];
            for (int c = 0; c < numClasses; c++) nc[j][c] += classCounts[i][c];
        }
        this.min = newMin;
        this.max = newMax;
        this.width = newWidth;
        this.binCounts = nb;
        this.classCounts = nc;
    }

    public void decay(double factor) {
        if (!(factor > 0.0 && factor <= 1.0)) throw new IllegalArgumentException("factor in (0,1]");
        for (int i = 0; i < b1; i++) {
            binCounts[i] = (int) Math.round(binCounts[i] * factor);
            for (int c = 0; c < numClasses; c++) {
                classCounts[i][c] = (int) Math.round(classCounts[i][c] * factor);
            }
        }
    }

    public long getUnderflow() { return underflow; }
    public long getOverflow()  { return overflow; }
}