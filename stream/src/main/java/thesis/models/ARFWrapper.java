package thesis.models;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import lombok.Getter;
import moa.classifiers.meta.AdaptiveRandomForest;
import thesis.selection.FeatureSelector;

import java.util.Arrays;
import java.util.Set;

public class ARFWrapper implements ModelWrapper {

    @Getter private final FeatureSelector selector;
    private final FeatureSpace space;
    @Getter private final int ensembleSize;
    @Getter private final double lambda;
    @Getter private final boolean resetOnSelectionChange;
    @Getter private final boolean useHardFilter;

    private AdaptiveRandomForest arf;
    private InstancesHeader reducedHeader;
    private int[] cachedSelection;

    private static final boolean DIAG = Boolean.getBoolean("thesis.diag");

    public ARFWrapper(FeatureSelector selector, InstancesHeader fullHeader) {
        this(selector, fullHeader, 10, 6.0, false, true);
    }

    public ARFWrapper(FeatureSelector selector, InstancesHeader fullHeader,
                      int ensembleSize, double lambda, boolean resetOnSelectionChange) {
        this(selector, fullHeader, ensembleSize, lambda, resetOnSelectionChange, true);
    }

    public ARFWrapper(FeatureSelector selector, InstancesHeader fullHeader,
                      int ensembleSize, double lambda,
                      boolean resetOnSelectionChange, boolean useHardFilter) {
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
            System.err.println("[ARFWrapper][WARN] useHardFilter=false → ARF sees FULL d features.");
        }
        rebuild();
    }

    private AdaptiveRandomForest newARF() {
        AdaptiveRandomForest a = new AdaptiveRandomForest();
        a.ensembleSizeOption.setValue(ensembleSize);
        try {
            a.getOptions().getOption('a').setValueViaCLIString(String.valueOf(lambda));
        } catch (Exception e) {
            throw new IllegalStateException("ARF option 'a' (lambda) not available", e);
        }
        a.prepareForUse();
        return a;
    }

    private void rebuild() {
        cachedSelection = selector.getCurrentSelection().clone();
        arf = newARF();
        reducedHeader = useHardFilter
                ? FilteredHeaderBuilder.build(space, cachedSelection, "_arf")
                : space.getHeader();
        arf.setModelContext(reducedHeader);
    }

    private void syncSelection() {
        int[] curr = selector.getCurrentSelection();
        if (Arrays.equals(curr, cachedSelection)) return;
        cachedSelection = curr.clone();
        if (useHardFilter) {
            reducedHeader = FilteredHeaderBuilder.build(space, cachedSelection, "_arf");
            if (resetOnSelectionChange) arf = newARF();
            arf.setModelContext(reducedHeader);
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
        return arf.getVotesForInstance(toModelInstance(full));
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
        Instance toModel = useHardFilter
                ? FilteredHeaderBuilder.filteredInstance(full, space, cachedSelection, reducedHeader)
                : full;
        toModel.setClassValue(classLabel);
        arf.trainOnInstance(toModel);
    }

    @Override public int[] getCurrentSelection() { return cachedSelection.clone(); }
    @Override public void reset() { rebuild(); }
    @Override public String name() {
        return "ARF(size=" + ensembleSize + ", lambda=" + lambda
                + ", hardFilter=" + useHardFilter + ") + " + selector.name();
    }
}