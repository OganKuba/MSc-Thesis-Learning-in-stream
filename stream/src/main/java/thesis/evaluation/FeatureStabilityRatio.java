package thesis.evaluation;

import java.util.HashSet;
import java.util.Set;

public class FeatureStabilityRatio {

    private int[] previous;
    private double sumRatios;
    private long updates;
    private double lastRatio = Double.NaN;

    public void update(int[] currentSelection) {
        if (currentSelection == null) return;
        if (previous == null) {
            previous = currentSelection.clone();
            return;
        }
        if (previous.length == 0) {
            previous = currentSelection.clone();
            return;
        }
        Set<Integer> oldSet = new HashSet<>();
        for (int v : previous) oldSet.add(v);
        int intersection = 0;
        Set<Integer> seen = new HashSet<>();
        for (int v : currentSelection) {
            if (seen.add(v) && oldSet.contains(v)) intersection++;
        }
        double ratio = (double) intersection / previous.length;
        sumRatios += ratio;
        updates++;
        lastRatio = ratio;
        previous = currentSelection.clone();
    }

    public double getLastRatio()    { return lastRatio; }
    public double getAverageRatio() { return updates == 0 ? Double.NaN : sumRatios / updates; }
    public long getUpdateCount()    { return updates; }

    public void reset() {
        previous = null;
        sumRatios = 0.0;
        updates = 0;
        lastRatio = Double.NaN;
    }
}