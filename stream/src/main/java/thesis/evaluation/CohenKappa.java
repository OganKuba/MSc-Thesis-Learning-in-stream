package thesis.evaluation;

import java.util.ArrayDeque;
import java.util.Deque;

public class CohenKappa {

    private final int numClasses;
    private final int windowSize;
    private final long[][] cm;
    private final long[] rowTotals;
    private final long[] colTotals;
    private final Deque<int[]> window;
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
        this.window = new ArrayDeque<>(windowSize);
    }

    public void update(int yTrue, int yPred) {
        if (yTrue < 0 || yTrue >= numClasses || yPred < 0 || yPred >= numClasses) return;
        if (window.size() == windowSize) {
            int[] old = window.pollFirst();
            int ot = old[0], op = old[1];
            cm[ot][op]--;
            rowTotals[ot]--;
            colTotals[op]--;
            total--;
            if (ot == op) correct--;
        }
        window.addLast(new int[]{yTrue, yPred});
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
        if (1.0 - pe < 1e-12) return 0.0;
        return (po - pe) / (1.0 - pe);
    }

    public double getAccuracy() {
        return total == 0 ? 0.0 : (double) correct / total;
    }

    public long getWindowCount() { return total; }
    public int getWindowSize()   { return windowSize; }

    public void reset() {
        for (int i = 0; i < numClasses; i++) {
            rowTotals[i] = 0;
            colTotals[i] = 0;
            for (int j = 0; j < numClasses; j++) cm[i][j] = 0;
        }
        window.clear();
        total = 0;
        correct = 0;
    }
}