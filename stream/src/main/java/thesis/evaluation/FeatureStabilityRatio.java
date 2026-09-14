package thesis.evaluation;

import java.util.BitSet;

public class FeatureStabilityRatio {

    private int[] previous;
    private double sumRatios;
    private long updates;
    private long changes;
    private double lastRatio = Double.NaN;
    private boolean lastChanged;

    public void update(int[] currentSelection) {
        if (currentSelection == null) return;
        int[] curr = currentSelection.clone();
        if (previous == null) {
            previous = curr;
            lastChanged = false;
            return;
        }
        int maxIdx = 0;
        for (int v : previous) if (v > maxIdx) maxIdx = v;
        for (int v : curr)     if (v > maxIdx) maxIdx = v;
        BitSet a = new BitSet(maxIdx + 1);
        BitSet b = new BitSet(maxIdx + 1);
        for (int v : previous) if (v >= 0) a.set(v);
        for (int v : curr)     if (v >= 0) b.set(v);
        BitSet inter = (BitSet) a.clone(); inter.and(b);
        int interC = inter.cardinality();
        int oldC = a.cardinality();
        double ratio = oldC == 0 ? (curr.length == 0 ? 1.0 : 0.0) : (double) interC / oldC;
        sumRatios += ratio;
        updates++;
        lastRatio = ratio;
        lastChanged = !arraysEqualAsSet(previous, curr);
        if (lastChanged) changes++;
        previous = curr;
    }

    private static boolean arraysEqualAsSet(int[] a, int[] b) {
        if (a.length != b.length) return false;
        BitSet sa = new BitSet();
        BitSet sb = new BitSet();
        for (int v : a) if (v >= 0) sa.set(v);
        for (int v : b) if (v >= 0) sb.set(v);
        return sa.equals(sb);
    }

    public double getLastRatio()    { return lastRatio; }
    public boolean wasLastChanged() { return lastChanged; }
    public double getAverageRatio() { return updates == 0 ? Double.NaN : sumRatios / updates; }
    public long getUpdateCount()    { return updates; }
    public long getChangeCount()    { return changes; }

    public void reset() {
        previous = null;
        sumRatios = 0.0;
        updates = 0;
        changes = 0;
        lastRatio = Double.NaN;
        lastChanged = false;
    }
}