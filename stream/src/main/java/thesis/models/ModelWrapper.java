package thesis.models;

import com.yahoo.labs.samoa.instances.Instance;
import thesis.selection.FeatureSelector;

import java.util.Set;

public interface ModelWrapper {

    double[] predictProba(Instance full);

    int predict(Instance full);

    void train(Instance full, int classLabel);

    void train(Instance full, int classLabel,
               boolean driftAlarm, Set<Integer> driftingFeatures);

    FeatureSelector getSelector();

    int[] getCurrentSelection();

    void reset();

    default String name() { return getClass().getSimpleName(); }
}