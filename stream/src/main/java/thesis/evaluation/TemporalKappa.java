package thesis.evaluation;

import java.util.ArrayDeque;
import java.util.Deque;

public class TemporalKappa {

    private final int windowSize;
    private final Deque<int[]> window;
    private long total;
    private long correct;
    private long noChangeCorrect;
    private int prevTrue = -1;

    public TemporalKappa(int windowSize) {
        if (windowSize < 1) throw new IllegalArgumentException("windowSize must be >= 1");
        this.windowSize = windowSize;
        this.window = new ArrayDeque<>(windowSize);
    }

    public void update(int yTrue, int yPred) {
        int yNc = (prevTrue < 0) ? yTrue : prevTrue;
        if (window.size() == windowSize) {
            int[] old = window.pollFirst();
            total--;
            if (old[0] == old[1]) correct--;
            if (old[0] == old[2]) noChangeCorrect--;
        }
        window.addLast(new int[]{yTrue, yPred, yNc});
        total++;
        if (yTrue == yPred) correct++;
        if (yTrue == yNc) noChangeCorrect++;
        prevTrue = yTrue;
    }

    public double getKappaTemporal() {
        if (total == 0) return 0.0;
        double po = (double) correct / total;
        double pnc = (double) noChangeCorrect / total;
        if (1.0 - pnc < 1e-12) return 0.0;
        return (po - pnc) / (1.0 - pnc);
    }

    public double getNoChangeAccuracy() {
        return total == 0 ? 0.0 : (double) noChangeCorrect / total;
    }

    public void reset() {
        window.clear();
        total = 0;
        correct = 0;
        noChangeCorrect = 0;
        prevTrue = -1;
    }
}