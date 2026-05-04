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

public class SRPWrapper implements ModelWrapper {

    private static final String[] ENSEMBLE_FIELD_CANDIDATES = {
            "ensemble", "learners", "baseLearners", "classifiers"
    };
    private static final String[] SUBSPACE_FIELD_CANDIDATES = {
            "subSpaceIndexes", "subspaceIndexes", "subSpace", "subspace",
            "indices", "subSpaceIndices", "subspaceIndices"
    };

    @Getter private final FeatureSelector selector;
    private final FeatureSpace space;
    @Getter private final int ensembleSize;
    @Getter private final double lambda;
    @Getter private final boolean resetOnSelectionChange;
    @Getter private final boolean useHardFilter;

    private StreamingRandomPatches srp;
    private InstancesHeader reducedHeader;
    private int[] cachedSelection;

    private static final boolean DIAG = Boolean.getBoolean("thesis.diag");

    public SRPWrapper(FeatureSelector selector, InstancesHeader fullHeader) {
        this(selector, fullHeader, 10, 6.0, false, true);
    }

    public SRPWrapper(FeatureSelector selector, InstancesHeader fullHeader,
                      int ensembleSize, double lambda,
                      boolean resetOnSelectionChange) {
        this(selector, fullHeader, ensembleSize, lambda, resetOnSelectionChange, true);
    }

    public SRPWrapper(FeatureSelector selector, InstancesHeader fullHeader,
                      int ensembleSize, double lambda,
                      boolean resetOnSelectionChange,
                      boolean useHardFilter) {
        if (selector == null) throw new IllegalArgumentException("selector must not be null");
        if (fullHeader == null) throw new IllegalArgumentException("fullHeader must not be null");
        if (!selector.isInitialized())
            throw new IllegalArgumentException("selector must be initialized before wrapping");
        if (ensembleSize < 1) throw new IllegalArgumentException("ensembleSize must be >= 1");
        if (lambda <= 0.0) throw new IllegalArgumentException("lambda must be > 0");
        this.selector = selector;
        this.space = new FeatureSpace(fullHeader);
        this.ensembleSize = ensembleSize;
        this.lambda = lambda;
        this.resetOnSelectionChange = resetOnSelectionChange;
        this.useHardFilter = useHardFilter;
        if (!useHardFilter && DIAG) {
            System.err.println("[SRPWrapper][WARN] useHardFilter=false → SRP sees FULL d features, "
                    + "FeatureSelector will NOT change model input. Use only for ablation.");
        }
        rebuild();
    }

    private StreamingRandomPatches newSRP() {
        StreamingRandomPatches s = new StreamingRandomPatches();
        s.ensembleSizeOption.setValue(ensembleSize);
        trySetCli(s, 'a', String.valueOf(lambda));
        trySetCli(s, 'o', "randompatches");
        s.prepareForUse();
        return s;
    }

    private static boolean trySetCli(StreamingRandomPatches s, char optChar, String value) {
        try {
            com.github.javacliparser.Option opt = s.getOptions().getOption(optChar);
            if (opt == null) return false;
            opt.setValueViaCLIString(value);
            return true;
        } catch (Exception e) { return false; }
    }

    private void rebuild() {
        cachedSelection = selector.getCurrentSelection().clone();
        srp = newSRP();
        reducedHeader = useHardFilter
                ? FilteredHeaderBuilder.build(space, cachedSelection, "_srp")
                : space.getHeader();
        srp.setModelContext(reducedHeader);
    }

    private void syncSelection() {
        int[] curr = selector.getCurrentSelection();
        if (Arrays.equals(curr, cachedSelection)) return;
        cachedSelection = curr.clone();
        if (useHardFilter) {
            reducedHeader = FilteredHeaderBuilder.build(space, cachedSelection, "_srp");
            if (resetOnSelectionChange) srp = newSRP();
            srp.setModelContext(reducedHeader);
        }
    }

    private Instance toModelInstance(Instance full) {
        return useHardFilter
                ? FilteredHeaderBuilder.filteredInstance(full, space, cachedSelection, reducedHeader)
                : full;
    }

    @Override
    public double[] predictProba(Instance full) {
        syncSelection();
        return srp.getVotesForInstance(toModelInstance(full));
    }

    @Override
    public int predict(Instance full) {
        double[] v = predictProba(full);
        if (v == null || v.length == 0) return 0;
        int best = 0;
        for (int i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    @Override
    public void train(Instance full, int classLabel) { train(full, classLabel, false, Set.of()); }

    @Override
    public void train(Instance full, int classLabel,
                      boolean driftAlarm, Set<Integer> driftingFeatures) {
        syncSelection();
        Instance toModel = toModelInstance(full);
        toModel.setClassValue(classLabel);
        srp.trainOnInstance(toModel);
    }

    public int[] getSubspaceIndices(int learnerIdx) {
        Object[] e = readEnsembleArray();
        if (e == null) return new int[0];
        if (learnerIdx < 0 || learnerIdx >= e.length)
            throw new IndexOutOfBoundsException("learnerIdx out of range: " + learnerIdx);
        return readSubspace(e[learnerIdx]);
    }

    public int[][] getAllSubspaceIndices() {
        Object[] e = readEnsembleArray();
        if (e == null) return new int[0][];
        int[][] out = new int[e.length][];
        for (int i = 0; i < e.length; i++) out[i] = readSubspace(e[i]);
        return out;
    }

    public int getActualEnsembleSize() {
        Object[] e = readEnsembleArray();
        return e == null ? 0 : e.length;
    }

    public void requireSubspaceField() {
        Object[] e = readEnsembleArray();
        if (e == null || e.length == 0)
            throw new IllegalStateException("SRP ensemble not available — train first");
        Class<?> c = e[0].getClass();
        while (c != null && c != Object.class) {
            for (Field f : c.getDeclaredFields()) if (f.getType() == int[].class) return;
            c = c.getSuperclass();
        }
        throw new IllegalStateException("No int[] subspace field on " + e[0].getClass().getName());
    }

    private Object[] readEnsembleArray() {
        for (String n : ENSEMBLE_FIELD_CANDIDATES) {
            Object v = readField(srp, n);
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
        for (String n : SUBSPACE_FIELD_CANDIDATES) {
            Object v = readField(learner, n);
            if (v instanceof int[] && ((int[]) v).length > 0) return ((int[]) v).clone();
            if (v instanceof Integer[]) {
                Integer[] b = (Integer[]) v;
                int[] o = new int[b.length];
                for (int i = 0; i < b.length; i++) o[i] = b[i];
                return o;
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
            } catch (NoSuchFieldException e) { c = c.getSuperclass(); }
            catch (IllegalAccessException e) { return null; }
        }
        return null;
    }

    public Instance buildFilteredInstance(Instance full) {
        syncSelection();
        return toModelInstance(full);
    }

    public InstancesHeader getReducedHeader() { syncSelection(); return reducedHeader; }
    public FeatureSpace getFeatureSpace() { return space; }

    @Override public int[] getCurrentSelection() { return cachedSelection.clone(); }
    @Override public void reset() { rebuild(); }
    @Override public String name() {
        return "SRP(size=" + ensembleSize + ", lambda=" + lambda
                + ", hardFilter=" + useHardFilter + ") + " + selector.name();
    }
}