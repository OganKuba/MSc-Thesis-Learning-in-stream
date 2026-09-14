package thesis.evaluation;

public class CohenKappa {

    private final int numClasses;
    private final int windowSize;
    private final long[][] cm;
    private final long[] rowTotals;
    private final long[] colTotals;
    private final int[] winTrue;
    private final int[] winPred;
    private int head;
    private int size;
    private long total;
    private long correct;

    public CohenKappa(int numClasses, int windowSize) {
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        if (windowSize < 1) throw new IllegalArgumentException("windowSize must be >= 1");
        this.numClasses = numClasses;
        this.windowSize = windowSize;
        this.cm = new long[numClasses][numClasses];
        this.rowTotals = new long[numClasses];
        this.colTotals = new long[numClasses];
        this.winTrue = new int[windowSize];
        this.winPred = new int[windowSize];
    }

    public void update(int yTrue, int yPred) {
        if (yTrue < 0 || yTrue >= numClasses)
            throw new IllegalArgumentException("yTrue out of range: " + yTrue);
        if (yPred < 0 || yPred >= numClasses)
            throw new IllegalArgumentException("yPred out of range: " + yPred);

        int writeIdx;
        if (size == windowSize) {
            int ot = winTrue[head], op = winPred[head];
            cm[ot][op]--;
            rowTotals[ot]--;
            colTotals[op]--;
            total--;
            if (ot == op) correct--;
            writeIdx = head;
            head = (head + 1) % windowSize;
        } else {
            writeIdx = (head + size) % windowSize;
            size++;
        }
        winTrue[writeIdx] = yTrue;
        winPred[writeIdx] = yPred;
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
    public long getWindowCount() { return total; }
    public int getWindowSize()   { return windowSize; }
    public int getNumClasses()   { return numClasses; }

    public void reset() {
        for (int i = 0; i < numClasses; i++) {
            rowTotals[i] = 0;
            colTotals[i] = 0;
            for (int j = 0; j < numClasses; j++) cm[i][j] = 0;
        }
        head = 0;
        size = 0;
        total = 0;
        correct = 0;
    }
}