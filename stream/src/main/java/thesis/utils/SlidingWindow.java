package thesis.utils;

public class SlidingWindow {

    private final int capacity;
    private final int numFeatures;
    private final double[][] data;
    private int head;
    private int size;

    public SlidingWindow(int capacity, int numFeatures) {
        if (capacity < 1) throw new IllegalArgumentException("capacity must be >= 1");
        if (numFeatures < 1) throw new IllegalArgumentException("numFeatures must be >= 1");
        this.capacity = capacity;
        this.numFeatures = numFeatures;
        this.data = new double[capacity][numFeatures];
        this.head = 0;
        this.size = 0;
    }

    public void add(double[] instance) {
        if (instance == null) {
            throw new IllegalArgumentException("instance must not be null");
        }
        if (instance.length != numFeatures) {
            throw new IllegalArgumentException(
                    "expected " + numFeatures + " features, got " + instance.length);
        }
        System.arraycopy(instance, 0, data[head], 0, numFeatures);
        head = (head + 1) % capacity;
        if (size < capacity) size++;
    }

    public double[][] getWindow() {
        double[][] out = new double[size][numFeatures];
        int start = (head - size + capacity) % capacity;
        for (int i = 0; i < size; i++) {
            System.arraycopy(data[(start + i) % capacity], 0, out[i], 0, numFeatures);
        }
        return out;
    }

    public int size() {
        return size;
    }

    public int capacity() {
        return capacity;
    }

    public int getNumFeatures() {
        return numFeatures;
    }

    public boolean isFull() {
        return size == capacity;
    }

    public void clear() {
        head = 0;
        size = 0;
    }
}