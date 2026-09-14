package thesis.evaluation;

public class CumulativeKappa {

    private final int numClasses;
    private final long[][] cm;
    private final long[] rowTotals;
    private final long[] colTotals;
    private long total;
    private long correct;

    public CumulativeKappa(int numClasses) {
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        this.numClasses = numClasses;
        this.cm = new long[numClasses][numClasses];
        this.rowTotals = new long[numClasses];
        this.colTotals = new long[numClasses];
    }

    public void update(int yTrue, int yPred) {
        if (yTrue < 0 || yTrue >= numClasses) return;
        if (yPred < 0 || yPred >= numClasses) return;
        cm[yTrue][yPred]++;
        rowTotals[yTrue]++;
        colTotals[yPred]++;
        total++;
        if (yTrue == yPred) correct++;
    }

    public double getKappa() {
        if (total == 0) return 0.0;
        double po = (double) correct / total;
        double pe = 0.0;
        double n = total;
        for (int i = 0; i < numClasses; i++) {
            pe += (rowTotals[i] / n) * (colTotals[i] / n);
        }
        double denom = 1.0 - pe;
        if (Math.abs(denom) < 1e-12) return po >= 1.0 - 1e-12 ? 1.0 : 0.0;
        double k = (po - pe) / denom;
        if (k > 1.0) k = 1.0;
        if (k < -1.0) k = -1.0;
        return k;
    }

    public double getAccuracy() { return total == 0 ? 0.0 : (double) correct / total; }
    public long getTotal() { return total; }
}
