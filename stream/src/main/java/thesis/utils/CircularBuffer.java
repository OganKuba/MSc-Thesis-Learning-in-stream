package thesis.utils;

import java.util.ArrayList;
import java.util.List;

public class CircularBuffer<T> {

    private final Object[] data;
    private final int capacity;
    private int head;
    private int size;

    public CircularBuffer(int capacity) {
        if (capacity < 1) {
            throw new IllegalArgumentException("capacity must be >= 1");
        }
        this.capacity = capacity;
        this.data = new Object[capacity];
        this.head = 0;
        this.size = 0;
    }

    public void add(T value) {
        data[head] = value;
        head = (head + 1) % capacity;
        if (size < capacity) size++;
    }

    @SuppressWarnings("unchecked")
    public T get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("index " + index + " out of [0, " + size + ")");
        }
        int start = (head - size + capacity) % capacity;
        return (T) data[(start + index) % capacity];
    }

    public List<T> getAll() {
        List<T> out = new ArrayList<>(size);
        for (int i = 0; i < size; i++) out.add(get(i));
        return out;
    }

    public int size() {
        return size;
    }

    public int capacity() {
        return capacity;
    }

    public boolean isFull() {
        return size == capacity;
    }

    public void clear() {
        for (int i = 0; i < capacity; i++) data[i] = null;
        head = 0;
        size = 0;
    }
}