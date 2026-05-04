package thesis.models;

import com.yahoo.labs.samoa.instances.Instance;
import lombok.Getter;
import thesis.selection.FeatureSelector;

import java.util.Arrays;
import java.util.Set;

public class NoChangeWrapper implements ModelWrapper {

    @Getter private final FeatureSelector selector;
    private final int numClasses;
    private int lastLabel = -1;

    public NoChangeWrapper(FeatureSelector selector, int numClasses) {
        if (selector == null) throw new IllegalArgumentException("selector must not be null");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        this.selector = selector;
        this.numClasses = numClasses;
    }

    @Override
    public double[] predictProba(Instance full) {
        double[] p = new double[numClasses];
        if (lastLabel < 0) {
            Arrays.fill(p, 1.0 / numClasses);
        } else {
            p[lastLabel] = 1.0;
        }
        return p;
    }

    @Override
    public int predict(Instance full) {
        return lastLabel < 0 ? 0 : lastLabel;
    }

    @Override
    public void train(Instance full, int classLabel) {
        if (classLabel >= 0 && classLabel < numClasses) lastLabel = classLabel;
    }

    @Override
    public void train(Instance full, int classLabel, boolean driftAlarm, Set<Integer> driftingFeatures) {
        train(full, classLabel);
    }

    @Override
    public int[] getCurrentSelection() { return selector.getCurrentSelection(); }

    @Override
    public void reset() { lastLabel = -1; }

    @Override
    public String name() { return "NoChange"; }
}