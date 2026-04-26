package thesis.models;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import lombok.Getter;
import moa.classifiers.meta.AdaptiveRandomForest;
import thesis.selection.FeatureSelector;

import java.util.Arrays;
import java.util.Set;

@Getter
public class ARFWrapper implements ModelWrapper {

    private final FeatureSelector selector;
    private final FeatureSpace space;
    private final int ensembleSize;
    private final double lambda;
    private final boolean resetOnSelectionChange;

    private AdaptiveRandomForest arf;
    private InstancesHeader reducedHeader;
    private int[] cachedSelection;

    public ARFWrapper(FeatureSelector selector, InstancesHeader fullHeader) {
        this(selector, fullHeader, 10, 6.0, false);
    }

    public ARFWrapper(FeatureSelector selector, InstancesHeader fullHeader,
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

    private AdaptiveRandomForest newARF() {
        AdaptiveRandomForest a = new AdaptiveRandomForest();
        a.ensembleSizeOption.setValue(ensembleSize);
        try {
            a.getOptions().getOption('a').setValueViaCLIString(String.valueOf(lambda));
        } catch (Exception ignored) { }
        a.prepareForUse();
        return a;
    }

    private void rebuild() {
        cachedSelection = selector.getCurrentSelection();
        reducedHeader = FilteredHeaderBuilder.build(space, cachedSelection, "_arf");
        arf = newARF();
        arf.setModelContext(reducedHeader);
    }

    private void syncSelection() {
        int[] curr = selector.getCurrentSelection();
        if (Arrays.equals(curr, cachedSelection)) return;
        cachedSelection = curr;
        reducedHeader = FilteredHeaderBuilder.build(space, cachedSelection, "_arf");
        if (resetOnSelectionChange) {
            arf = newARF();
        }
        arf.setModelContext(reducedHeader);
    }

    @Override
    public double[] predictProba(Instance full) {
        syncSelection();
        Instance reduced = FilteredHeaderBuilder.filteredInstance(
                full, space, cachedSelection, reducedHeader);
        return arf.getVotesForInstance(reduced);
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
        arf.trainOnInstance(reduced);
        selector.update(space.extractFeatures(full), classLabel,
                driftAlarm, driftingFeatures == null ? Set.of() : driftingFeatures);
    }

    @Override
    public FeatureSelector getSelector()  { return selector; }
    @Override
    public int[] getCurrentSelection()    { return cachedSelection.clone(); }
    @Override
    public void reset()                   { rebuild(); }

    public AdaptiveRandomForest getARF()  { return arf; }

    @Override
    public String name() {
        return "ARF(size=" + ensembleSize + ", lambda=" + lambda + ") + " + selector.name();
    }
}