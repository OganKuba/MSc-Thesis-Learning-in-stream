package thesis.evaluation;

public class PrequentialAccuracy {

    private final int windowSize;
    private final boolean[] win;
    private int head;
    private int size;
    private long correct;

    public PrequentialAccuracy(int windowSize) {
        if (windowSize < 1) throw new IllegalArgumentException("windowSize must be >= 1");
        this.windowSize = windowSize;
        this.win = new boolean[windowSize];
    }

    public void update(int yTrue, int yPred) {
        boolean ok = (yTrue == yPred);
        int writeIdx;
        if (size == windowSize) {
            if (win[head]) correct--;
            writeIdx = head;
            head = (head + 1) % windowSize;
        } else {
            writeIdx = (head + size) % windowSize;
            size++;
        }
        win[writeIdx] = ok;
        if (ok) correct++;
    }

    public double getAccuracy() { return size == 0 ? 0.0 : (double) correct / size; }
    public int getCount()       { return size; }
    public int getWindowSize()  { return windowSize; }

    public void reset() {
        head = 0;
        size = 0;
        correct = 0;
        for (int i = 0; i < windowSize; i++) win[i] = false;
    }
}