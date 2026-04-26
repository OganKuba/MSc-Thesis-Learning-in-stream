package thesis.models;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import lombok.Getter;
import moa.classifiers.trees.HoeffdingTree;
import thesis.selection.FeatureSelector;

import java.util.Arrays;
import java.util.Set;

@Getter
public class HoeffdingTreeWrapper implements ModelWrapper {

    private final FeatureSelector selector;
    private final FeatureSpace space;
    private final int gracePeriod;
    private final double splitConfidence;
    private final boolean resetOnSelectionChange;

    private HoeffdingTree tree;
    private InstancesHeader reducedHeader;
    private int[] cachedSelection;

    public HoeffdingTreeWrapper(FeatureSelector selector, InstancesHeader fullHeader) {
        this(selector, fullHeader, 200, 0.01, false);
    }

    public HoeffdingTreeWrapper(FeatureSelector selector, InstancesHeader fullHeader,
                                int gracePeriod, double splitConfidence,
                                boolean resetOnSelectionChange) {
        if (selector == null) throw new IllegalArgumentException("selector must not be null");
        if (fullHeader == null) throw new IllegalArgumentException("fullHeader must not be null");
        if (!selector.isInitialized()) {
            throw new IllegalArgumentException("selector must be initialized before wrapping");
        }
        if (gracePeriod < 1) throw new IllegalArgumentException("gracePeriod must be >= 1");
        if (!(splitConfidence > 0.0 && splitConfidence < 1.0)) {
            throw new IllegalArgumentException("splitConfidence must be in (0,1)");
        }
        this.selector = selector;
        this.space = new FeatureSpace(fullHeader);
        this.gracePeriod = gracePeriod;
        this.splitConfidence = splitConfidence;
        this.resetOnSelectionChange = resetOnSelectionChange;
        rebuild();
    }

    private HoeffdingTree newTree() {
        HoeffdingTree t = new HoeffdingTree();
        t.gracePeriodOption.setValue(gracePeriod);
        t.splitConfidenceOption.setValue(splitConfidence);
        t.prepareForUse();
        return t;
    }

    private void rebuild() {
        cachedSelection = selector.getCurrentSelection();
        reducedHeader = FilteredHeaderBuilder.build(space, cachedSelection, "_ht");
        tree = newTree();
        tree.setModelContext(reducedHeader);
    }

    private void syncSelection() {
        int[] curr = selector.getCurrentSelection();
        if (Arrays.equals(curr, cachedSelection)) return;
        cachedSelection = curr;
        reducedHeader = FilteredHeaderBuilder.build(space, cachedSelection, "_ht");
        if (resetOnSelectionChange) {
            tree = newTree();
        }
        tree.setModelContext(reducedHeader);
    }

    @Override
    public double[] predictProba(Instance full) {
        syncSelection();
        Instance reduced = FilteredHeaderBuilder.filteredInstance(
                full, space, cachedSelection, reducedHeader);
        return tree.getVotesForInstance(reduced);
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
        tree.trainOnInstance(reduced);
        selector.update(space.extractFeatures(full), classLabel,
                driftAlarm, driftingFeatures == null ? Set.of() : driftingFeatures);
    }

    @Override
    public FeatureSelector getSelector()  { return selector; }
    @Override
    public int[] getCurrentSelection()    { return cachedSelection.clone(); }
    @Override
    public void reset()                   { rebuild(); }

    @Override
    public String name() {
        return "HoeffdingTree(gp=" + gracePeriod + ", sc=" + splitConfidence + ") + " + selector.name();
    }
}