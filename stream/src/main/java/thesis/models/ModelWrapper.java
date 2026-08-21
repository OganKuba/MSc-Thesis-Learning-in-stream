package thesis.models;

import com.yahoo.labs.samoa.instances.Instance;
import thesis.selection.FeatureSelector;

import java.util.Set;

public interface  ModelWrapper {

    double[] predictProba(Instance full);

    int predict(Instance full);

    void train(Instance full, int classLabel);

    void train(Instance full, int classLabel,
               boolean driftAlarm, Set<Integer> driftingFeatures);

    FeatureSelector getSelector();

    int[] getCurrentSelection();

    void reset();

    /**
     * Deep size of the learned model in bytes, for the RAM-Hours metric, or
     * {@link ModelSize#UNAVAILABLE} when the {@code sizeofag} agent is not loaded.
     * Implementations must report the <b>model only</b> — not the selector, header or JVM heap.
     */
    default long modelByteSize() { return ModelSize.UNAVAILABLE; }

    default String name() { return getClass().getSimpleName(); }
}