package thesis.models;

import java.util.Arrays;

public final class DriftActionSummary {

    public enum Action { KEEP, SURGICAL, FULL, NO_REPLACEMENT }

    private final Action[] perLearner;
    private final int[] overlapCounts;
    private final int[] subspaceSizes;
    private int kept;
    private int surgical;
    private int full;
    private int noReplacement;

    public DriftActionSummary(int ensembleSize) {
        this.perLearner = new Action[ensembleSize];
        this.overlapCounts = new int[ensembleSize];
        this.subspaceSizes = new int[ensembleSize];
        Arrays.fill(perLearner, Action.KEEP);
    }

    void record(int learnerIdx, Action action, int overlap, int subspaceSize) {
        perLearner[learnerIdx] = action;
        overlapCounts[learnerIdx] = overlap;
        subspaceSizes[learnerIdx] = subspaceSize;
        switch (action) {
            case KEEP:           kept++; break;
            case SURGICAL:       surgical++; break;
            case FULL:           full++; break;
            case NO_REPLACEMENT: noReplacement++; break;
        }
    }

    public Action[] getPerLearner()  { return perLearner.clone(); }
    public int[] getOverlapCounts()  { return overlapCounts.clone(); }
    public int[] getSubspaceSizes()  { return subspaceSizes.clone(); }
    public int getKeptCount()        { return kept; }
    public int getSurgicalCount()    { return surgical; }
    public int getFullCount()        { return full; }
    public int getNoReplacementCount() { return noReplacement; }
    public int getEnsembleSize()     { return perLearner.length; }

    @Override
    public String toString() {
        return "DriftActionSummary{ensemble=" + perLearner.length +
                ", keep=" + kept + ", surgical=" + surgical +
                ", full=" + full + ", noReplacement=" + noReplacement +
                ", actions=" + Arrays.toString(perLearner) +
                ", overlaps=" + Arrays.toString(overlapCounts) + "}";
    }
}