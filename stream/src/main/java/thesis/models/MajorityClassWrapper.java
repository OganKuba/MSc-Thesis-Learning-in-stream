package thesis.models;

import com.yahoo.labs.samoa.instances.Instance;
import lombok.Getter;
import thesis.selection.FeatureSelector;

import java.util.Arrays;
import java.util.Set;

public class MajorityClassWrapper implements ModelWrapper {

    @Getter private final FeatureSelector selector;
    private final int numClasses;
    private final long[] counts;
    private long total;

    public MajorityClassWrapper(FeatureSelector selector, int numClasses) {
        if (selector == null) throw new IllegalArgumentException("selector must not be null");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        this.selector = selector;
        this.numClasses = numClasses;
        this.counts = new long[numClasses];
    }

    @Override
    public double[] predictProba(Instance full) {
        double[] p = new double[numClasses];
        if (total == 0) {
            Arrays.fill(p, 1.0 / numClasses);
            return p;
        }
        double inv = 1.0 / total;
        for (int i = 0; i < numClasses; i++) p[i] = counts[i] * inv;
        return p;
    }

    @Override
    public int predict(Instance full) {
        int best = 0;
        long bestCount = counts[0];
        for (int i = 1; i < numClasses; i++) {
            if (counts[i] > bestCount) { bestCount = counts[i]; best = i; }
        }
        return best;
    }

    @Override
    public void train(Instance full, int classLabel) {
        if (classLabel >= 0 && classLabel < numClasses) {
            counts[classLabel]++;
            total++;
        }
    }

    @Override
    public void train(Instance full, int classLabel, boolean driftAlarm, Set<Integer> driftingFeatures) {
        train(full, classLabel);
    }

    @Override
    public int[] getCurrentSelection() { return selector.getCurrentSelection(); }

    @Override
    public void reset() {
        Arrays.fill(counts, 0);
        total = 0;
    }

    @Override
    public String name() { return "MajorityClass"; }
}