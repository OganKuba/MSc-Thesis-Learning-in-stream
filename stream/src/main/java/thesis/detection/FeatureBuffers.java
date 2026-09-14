package thesis.detection;

public class FeatureBuffers {

    private final int numFeatures;
    private final int windowSize;
    private final double[][] rolling;
    private int rollingHead;
    private int rollingSize;

    private double[][] reference;
    private final double[][] post;
    private int postSize;

    public FeatureBuffers(int numFeatures, int windowSize) {
        this.numFeatures = numFeatures;
        this.windowSize = windowSize;
        this.rolling = new double[numFeatures][windowSize];
        this.post = new double[numFeatures][windowSize];
        this.rollingHead = 0;
        this.rollingSize = 0;
        this.postSize = 0;
        this.reference = null;
    }

    public void pushRolling(double[] values) {
        int slot = (rollingHead + rollingSize) % windowSize;
        if (rollingSize < windowSize) {
            for (int f = 0; f < numFeatures; f++) rolling[f][slot] = values[f];
            rollingSize++;
        } else {
            for (int f = 0; f < numFeatures; f++) rolling[f][rollingHead] = values[f];
            rollingHead = (rollingHead + 1) % windowSize;
        }
    }

    public boolean isRollingFull() { return rollingSize == windowSize; }

    public void snapshotReferenceFromRolling() {
        if (rollingSize < windowSize) {
            reference = null;
            return;
        }
        double[][] snap = new double[numFeatures][windowSize];
        for (int f = 0; f < numFeatures; f++) {
            for (int i = 0; i < windowSize; i++) {
                snap[f][i] = rolling[f][(rollingHead + i) % windowSize];
            }
        }
        reference = snap;
    }

    public void pushPost(double[] values) {
        if (postSize >= windowSize) return;
        for (int f = 0; f < numFeatures; f++) post[f][postSize] = values[f];
        postSize++;
    }

    public void clearPost() { postSize = 0; }

    public double[][] getReference() { return reference; }

    public double[][] getPost() {
        if (postSize < windowSize) return null;
        double[][] copy = new double[numFeatures][windowSize];
        for (int f = 0; f < numFeatures; f++) {
            System.arraycopy(post[f], 0, copy[f], 0, windowSize);
        }
        return copy;
    }

    public void promotePostToReference() {
        if (postSize < windowSize) return;
        double[][] snap = new double[numFeatures][windowSize];
        for (int f = 0; f < numFeatures; f++) {
            System.arraycopy(post[f], 0, snap[f], 0, windowSize);
        }
        reference = snap;
        rollingHead = 0;
        rollingSize = windowSize;
        for (int f = 0; f < numFeatures; f++) {
            System.arraycopy(post[f], 0, rolling[f], 0, windowSize);
        }
        postSize = 0;
    }

    public void reset() {
        reference = null;
        rollingHead = 0;
        rollingSize = 0;
        postSize = 0;
    }
}