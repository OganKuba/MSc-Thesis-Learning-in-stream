package thesis.models;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import lombok.Getter;
import moa.classifiers.meta.StreamingRandomPatches;
import thesis.selection.FeatureSelector;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.Set;

@Getter
public class SRPWrapper implements ModelWrapper {

    private static final String[] ENSEMBLE_FIELD_CANDIDATES = {
            "ensemble", "learners", "baseLearners", "classifiers"
    };

    private static final String[] SUBSPACE_FIELD_CANDIDATES = {
            "subSpaceIndexes", "subspaceIndexes", "subSpace", "subspace",
            "indices", "subSpaceIndices", "subspaceIndices"
    };

    private final FeatureSelector selector;
    private final FeatureSpace space;
    private final int ensembleSize;
    private final double lambda;
    private final boolean resetOnSelectionChange;

    private StreamingRandomPatches srp;
    private InstancesHeader reducedHeader;
    private int[] cachedSelection;

    public SRPWrapper(FeatureSelector selector, InstancesHeader fullHeader) {
        this(selector, fullHeader, 10, 6.0, false);
    }

    public SRPWrapper(FeatureSelector selector, InstancesHeader fullHeader,
                      int ensembleSize, double lambda,
                      boolean resetOnSelectionChange) {
        if (selector == null) throw new IllegalArgumentException("selector must not be null");
        if (fullHeader == null) throw new IllegalArgumentException("fullHeader must not be null");
        if (!selector.isInitialized()) {
            throw new IllegalArgumentException("selector must be initialized before wrapping");
        }
        if (ensembleSize < 1) throw new IllegalArgumentException("ensembleSize must be >= 1");
        if (lambda <= 0.0) throw new IllegalArgumentException("lambda must be > 0");
        this.selector = selector;
        this.space = new FeatureSpace(fullHeader);
        this.ensembleSize = ensembleSize;
        this.lambda = lambda;
        this.resetOnSelectionChange = resetOnSelectionChange;
        rebuild();
    }

    private StreamingRandomPatches newSRP() {
        StreamingRandomPatches s = new StreamingRandomPatches();
        s.ensembleSizeOption.setValue(ensembleSize);
        trySetCli(s, 'a', String.valueOf(lambda));
        s.prepareForUse();
        return s;
    }

    private static void trySetCli(StreamingRandomPatches s, char optChar, String value) {
        try {
            s.getOptions().getOption(optChar).setValueViaCLIString(value);
        } catch (Exception ignored) { }
    }

    private void rebuild() {
        cachedSelection = selector.getCurrentSelection();
        reducedHeader = FilteredHeaderBuilder.build(space, cachedSelection, "_srp");
        srp = newSRP();
        srp.setModelContext(reducedHeader);
    }

    private void syncSelection() {
        int[] curr = selector.getCurrentSelection();
        if (Arrays.equals(curr, cachedSelection)) return;
        cachedSelection = curr;
        reducedHeader = FilteredHeaderBuilder.build(space, cachedSelection, "_srp");
        if (resetOnSelectionChange) {
            srp = newSRP();
        }
        srp.setModelContext(reducedHeader);
    }

    @Override
    public double[] predictProba(Instance full) {
        syncSelection();
        Instance reduced = FilteredHeaderBuilder.filteredInstance(
                full, space, cachedSelection, reducedHeader);
        return srp.getVotesForInstance(reduced);
    }

    @Override
    public int predict(Instance full) {
        double[] votes = predictProba(full);
        int best = 0;
        for (int i = 1; i < votes.length; i++) if (votes[i] > votes[best]) best = i;
        return best;
    }

    @Override
    public void train(Instance full, int classLabel) {
        train(full, classLabel, false, Set.of());
    }

    @Override
    public void train(Instance full, int classLabel,
                      boolean driftAlarm, Set<Integer> driftingFeatures) {
        syncSelection();
        Instance reduced = FilteredHeaderBuilder.filteredInstance(
                full, space, cachedSelection, reducedHeader);
        reduced.setClassValue(classLabel);
        srp.trainOnInstance(reduced);
        selector.update(space.extractFeatures(full), classLabel,
                driftAlarm, driftingFeatures == null ? Set.of() : driftingFeatures);
    }

    public int[] getSubspaceIndices(int learnerIdx) {
        Object[] ensemble = readEnsembleArray();
        if (ensemble == null) return new int[0];
        if (learnerIdx < 0 || learnerIdx >= ensemble.length) {
            throw new IndexOutOfBoundsException("learnerIdx out of range: " + learnerIdx);
        }
        return readSubspace(ensemble[learnerIdx]);
    }

    public int[][] getAllSubspaceIndices() {
        Object[] ensemble = readEnsembleArray();
        if (ensemble == null) return new int[0][];
        int[][] out = new int[ensemble.length][];
        for (int i = 0; i < ensemble.length; i++) {
            out[i] = readSubspace(ensemble[i]);
        }
        return out;
    }

    public int getActualEnsembleSize() {
        Object[] ensemble = readEnsembleArray();
        return ensemble == null ? 0 : ensemble.length;
    }

    public void requireSubspaceField() {
        Object[] ensemble = readEnsembleArray();
        if (ensemble == null || ensemble.length == 0) {
            throw new IllegalStateException(
                    "SRP ensemble not available — model may need a few train calls first");
        }
        int[] sub = readSubspace(ensemble[0]);
        if (sub.length == 0) {
            throw new IllegalStateException(
                    "Could not locate subspace indices field on " + ensemble[0].getClass().getName() +
                            " — tried: " + Arrays.toString(SUBSPACE_FIELD_CANDIDATES));
        }
    }

    private Object[] readEnsembleArray() {
        for (String name : ENSEMBLE_FIELD_CANDIDATES) {
            Object v = readField(srp, name);
            if (v == null) continue;
            if (v.getClass().isArray()) {
                int len = Array.getLength(v);
                Object[] out = new Object[len];
                for (int i = 0; i < len; i++) out[i] = Array.get(v, i);
                return out;
            }
        }
        return null;
    }

    private static int[] readSubspace(Object learner) {
        if (learner == null) return new int[0];
        for (String name : SUBSPACE_FIELD_CANDIDATES) {
            Object v = readField(learner, name);
            if (v instanceof int[]) {
                return ((int[]) v).clone();
            }
            if (v instanceof Integer[]) {
                Integer[] boxed = (Integer[]) v;
                int[] out = new int[boxed.length];
                for (int i = 0; i < boxed.length; i++) out[i] = boxed[i];
                return out;
            }
        }
        return new int[0];
    }

    private static Object readField(Object obj, String name) {
        Class<?> c = obj.getClass();
        while (c != null && c != Object.class) {
            try {
                Field f = c.getDeclaredField(name);
                f.setAccessible(true);
                return f.get(obj);
            } catch (NoSuchFieldException e) {
                c = c.getSuperclass();
            } catch (IllegalAccessException e) {
                return null;
            }
        }
        return null;
    }

    public Instance buildFilteredInstance(Instance full) {
        syncSelection();
        return FilteredHeaderBuilder.filteredInstance(full, space, cachedSelection, reducedHeader);
    }

    public InstancesHeader getReducedHeader() {
        syncSelection();
        return reducedHeader;
    }

    public FeatureSpace getFeatureSpace() {
        return space;
    }

    @Override
    public FeatureSelector getSelector()  { return selector; }
    @Override
    public int[] getCurrentSelection()    { return cachedSelection.clone(); }
    @Override
    public void reset()                   { rebuild(); }

    public StreamingRandomPatches getSRP() { return srp; }
    public int getEnsembleSize()           { return ensembleSize; }
    public double getLambda()              { return lambda; }

    @Override
    public String name() {
        return "SRP(size=" + ensembleSize + ", lambda=" + lambda + ") + " + selector.name();
    }
}