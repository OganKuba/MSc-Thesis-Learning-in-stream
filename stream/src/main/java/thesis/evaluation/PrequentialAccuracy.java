package thesis.evaluation;

import java.util.ArrayDeque;
import java.util.Deque;

public class PrequentialAccuracy {

    private final int windowSize;
    private final Deque<Boolean> window;
    private long correct;

    public PrequentialAccuracy(int windowSize) {
        if (windowSize < 1) throw new IllegalArgumentException("windowSize must be >= 1");
        this.windowSize = windowSize;
        this.window = new ArrayDeque<>(windowSize);
    }

    public void update(int yTrue, int yPred) {
        boolean ok = (yTrue == yPred);
        if (window.size() == windowSize) {
            if (Boolean.TRUE.equals(window.pollFirst())) correct--;
        }
        window.addLast(ok);
        if (ok) correct++;
    }

    public double getAccuracy() {
        return window.isEmpty() ? 0.0 : (double) correct / window.size();
    }

    public int getCount() { return window.size(); }

    public void reset() {
        window.clear();
        correct = 0;
    }
}