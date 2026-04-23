package thesis.discretization;

import lombok.Getter;

@Getter
public class Layer1Histogram {

    private final int b1;
    private final int numClasses;

    private double min;
    private double max;
    private double width;
    private final int[] binCounts;
    private final int[][] classCounts;

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
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        if (mn == mx) mx = mn + 1.0;
        double margin = (mx - mn) * 0.05;
        Layer1Histogram h = new Layer1Histogram(b1, numClasses, mn - margin, mx + margin);
        for (int i = 0; i < values.length; i++) {
            h.update(values[i], classes[i]);
        }
        return h;
    }

    public void update(double value, int classLabel) {
        int idx = bin(value);
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
}