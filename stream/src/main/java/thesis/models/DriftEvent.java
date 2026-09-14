package thesis.models;

import java.util.Set;

public final class DriftEvent {

    public final long instanceIdx;
    public final boolean alarm;
    public final Set<Integer> driftingFeatures;
    public final DriftActionSummary summary;
    public final double[] importanceSnapshot;
    public final double[] learnerWeightsSnapshot;

    public DriftEvent(long instanceIdx, boolean alarm, Set<Integer> driftingFeatures,
                      DriftActionSummary summary,
                      double[] importanceSnapshot, double[] learnerWeightsSnapshot) {
        this.instanceIdx = instanceIdx;
        this.alarm = alarm;
        this.driftingFeatures = driftingFeatures;
        this.summary = summary;
        this.importanceSnapshot = importanceSnapshot;
        this.learnerWeightsSnapshot = learnerWeightsSnapshot;
    }

    @Override
    public String toString() {
        return "DriftEvent{i=" + instanceIdx + ", alarm=" + alarm
                + ", drifting=" + driftingFeatures
                + ", " + (summary == null ? "no-summary" : summary.toString()) + "}";
    }
}
