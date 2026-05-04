package thesis.evaluation;

public class TemporalKappa {

    private final int windowSize;
    private final int[] winTrue;
    private final int[] winPred;
    private final int[] winNc;
    private int head;
    private int size;
    private long total;
    private long correct;
    private long noChangeCorrect;
    private int prevTrue = Integer.MIN_VALUE;

    public TemporalKappa(int windowSize) {
        if (windowSize < 1) throw new IllegalArgumentException("windowSize must be >= 1");
        this.windowSize = windowSize;
        this.winTrue = new int[windowSize];
        this.winPred = new int[windowSize];
        this.winNc   = new int[windowSize];
    }

    public void update(int yTrue, int yPred) {
        if (yTrue < 0) throw new IllegalArgumentException("yTrue must be >= 0");
        if (yPred < 0) throw new IllegalArgumentException("yPred must be >= 0");
        if (prevTrue == Integer.MIN_VALUE) {
            prevTrue = yTrue;
            return;
        }
        int yNc = prevTrue;

        int writeIdx;
        if (size == windowSize) {
            int ot = winTrue[head], op = winPred[head], onc = winNc[head];
            total--;
            if (ot == op)  correct--;
            if (ot == onc) noChangeCorrect--;
            writeIdx = head;
            head = (head + 1) % windowSize;
        } else {
            writeIdx = (head + size) % windowSize;
            size++;
        }
        winTrue[writeIdx] = yTrue;
        winPred[writeIdx] = yPred;
        winNc[writeIdx]   = yNc;
        total++;
        if (yTrue == yPred) correct++;
        if (yTrue == yNc)   noChangeCorrect++;
        prevTrue = yTrue;
    }

    public double getKappaTemporal() {
        if (total == 0) return 0.0;
        double po  = (double) correct / total;
        double pnc = (double) noChangeCorrect / total;
        double denom = 1.0 - pnc;
        if (Math.abs(denom) < 1e-12) return po >= 1.0 - 1e-12 ? 1.0 : 0.0;
        double k = (po - pnc) / denom;
        if (k > 1.0) k = 1.0;
        if (k < -1.0) k = -1.0;
        return k;
    }

    public double getNoChangeAccuracy() {
        return total == 0 ? 0.0 : (double) noChangeCorrect / total;
    }

    public long getWindowCount() { return total; }
    public int getWindowSize()   { return windowSize; }

    public void reset() {
        head = 0;
        size = 0;
        total = 0;
        correct = 0;
        noChangeCorrect = 0;
        prevTrue = Integer.MIN_VALUE;
    }
}