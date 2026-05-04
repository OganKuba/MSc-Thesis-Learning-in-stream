package thesis.models;

import lombok.Getter;

import java.util.Arrays;

@Getter
public final class DriftActionSummary {

    public enum Action { KEEP, SURGICAL, FULL, NO_REPLACEMENT }

    private final Action[] perLearner;
    private final int[] overlapCounts;
    private final int[] subspaceSizes;
    private final int[] swapCounts;
    private int keptCount;
    private int surgicalCount;
    private int fullCount;
    private int noReplacementCount;

    public DriftActionSummary(int ensembleSize) {
        if (ensembleSize < 0) throw new IllegalArgumentException("ensembleSize must be >= 0");
        this.perLearner = new Action[ensembleSize];
        this.overlapCounts = new int[ensembleSize];
        this.subspaceSizes = new int[ensembleSize];
        this.swapCounts = new int[ensembleSize];
        Arrays.fill(perLearner, Action.KEEP);
    }

    void record(int learnerIdx, Action action, int overlap, int subspaceSize, int swaps) {
        if (learnerIdx < 0 || learnerIdx >= perLearner.length) {
            throw new IndexOutOfBoundsException("learnerIdx=" + learnerIdx);
        }
        if (perLearner[learnerIdx] != Action.KEEP || overlapCounts[learnerIdx] != 0) {
            throw new IllegalStateException("learner " + learnerIdx + " already recorded");
        }
        perLearner[learnerIdx] = action;
        overlapCounts[learnerIdx] = overlap;
        subspaceSizes[learnerIdx] = subspaceSize;
        swapCounts[learnerIdx] = swaps;
        switch (action) {
            case KEEP:           keptCount++; break;
            case SURGICAL:       surgicalCount++; break;
            case FULL:           fullCount++; break;
            case NO_REPLACEMENT: noReplacementCount++; break;
        }
    }

    public Action[] getPerLearner()  { return perLearner.clone(); }
    public int[] getOverlapCounts()  { return overlapCounts.clone(); }
    public int[] getSubspaceSizes()  { return subspaceSizes.clone(); }
    public int[] getSwapCounts()     { return swapCounts.clone(); }
    public int getEnsembleSize()     { return perLearner.length; }

    @Override
    public String toString() {
        return "DriftActionSummary{ensemble=" + perLearner.length +
                ", keep=" + keptCount + ", surgical=" + surgicalCount +
                ", full=" + fullCount + ", noReplacement=" + noReplacementCount +
                ", actions=" + Arrays.toString(perLearner) +
                ", overlaps=" + Arrays.toString(overlapCounts) +
                ", swaps=" + Arrays.toString(swapCounts) + "}";
    }
}