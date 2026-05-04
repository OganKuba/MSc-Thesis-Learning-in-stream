package thesis.models;

import lombok.Getter;
import java.util.Arrays;
import java.util.Set;

@Getter
public class FeatureImportance {

    private final int numFeatures;
    private final double w1;
    private final double w2;
    private final double epsilon;
    private final boolean normalizeInputs;

    private double[] miScores;
    private double[] ksStatistics;
    private double[] importance;
    private long updates;
    private long degenerateUniformFallbacks;
    private long projectionOutOfRange;

    public FeatureImportance(int numFeatures) { this(numFeatures, 0.7, 0.3, 1e-6, true); }

    public FeatureImportance(int numFeatures, double w1, double w2, double epsilon, boolean normalizeInputs) {
        if (numFeatures < 1) throw new IllegalArgumentException("numFeatures must be >= 1");
        if (w1 < 0.0 || w2 < 0.0) throw new IllegalArgumentException("weights must be non-negative");
        if (w1 + w2 == 0.0) throw new IllegalArgumentException("at least one of w1, w2 must be > 0");
        if (epsilon <= 0.0) throw new IllegalArgumentException("epsilon must be > 0");
        this.numFeatures = numFeatures;
        this.w1 = w1; this.w2 = w2; this.epsilon = epsilon; this.normalizeInputs = normalizeInputs;
        this.miScores = new double[numFeatures];
        this.ksStatistics = new double[numFeatures];
        this.importance = new double[numFeatures];
        Arrays.fill(this.importance, 1.0 / numFeatures);
    }

    public void update(double[] miScores, double[] ksStatistics) {
        if (miScores == null || miScores.length != numFeatures)
            throw new IllegalArgumentException("miScores length");
        if (ksStatistics == null || ksStatistics.length != numFeatures)
            throw new IllegalArgumentException("ksStatistics length");
        for (int i = 0; i < numFeatures; i++) {
            if (!Double.isFinite(miScores[i])) throw new IllegalArgumentException("miScores[" + i + "] not finite");
            if (!Double.isFinite(ksStatistics[i])) throw new IllegalArgumentException("ksStatistics[" + i + "] not finite");
        }
        double[] mi = normalizeInputs ? maxNormalize(miScores) : clampNonNeg(miScores);
        double[] stab = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            double ks = Math.max(0.0, ksStatistics[i]);
            stab[i] = 1.0 / (ks + epsilon);
        }
        if (normalizeInputs) stab = maxNormalize(stab);

        double sumImp = 0.0;
        double[] imp = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            imp[i] = w1 * mi[i] + w2 * stab[i];
            if (imp[i] < 0.0) imp[i] = 0.0;
            sumImp += imp[i];
        }
        if (sumImp <= 0.0 || !Double.isFinite(sumImp)) {
            Arrays.fill(imp, 1.0 / numFeatures);
            degenerateUniformFallbacks++;
        } else {
            for (int i = 0; i < numFeatures; i++) imp[i] /= sumImp;
        }
        this.miScores = miScores.clone();
        this.ksStatistics = ksStatistics.clone();
        this.importance = imp;
        this.updates++;
    }

    public void boost(Set<Integer> features, double factor) {
        if (features == null || features.isEmpty()) return;
        if (!(factor > 0.0) || !Double.isFinite(factor))
            throw new IllegalArgumentException("factor must be > 0");
        double sum = 0.0;
        for (int i = 0; i < numFeatures; i++) {
            if (features.contains(i)) importance[i] *= factor;
            sum += importance[i];
        }
        if (sum <= 0.0 || !Double.isFinite(sum)) {
            Arrays.fill(importance, 1.0 / numFeatures);
            degenerateUniformFallbacks++;
            return;
        }
        for (int i = 0; i < numFeatures; i++) importance[i] /= sum;
    }

    public void updateMIOnly(double[] miScores)     { update(miScores, this.ksStatistics); }
    public void updateKSOnly(double[] ksStatistics) { update(this.miScores, ksStatistics); }

    public void resetToUniform() {
        Arrays.fill(this.importance, 1.0 / numFeatures);
        Arrays.fill(this.miScores, 0.0);
        Arrays.fill(this.ksStatistics, 0.0);
    }

    private static double[] maxNormalize(double[] x) {
        double max = 0.0;
        for (double v : x) if (v > max) max = v;
        double[] out = new double[x.length];
        if (max <= 0.0) return out;
        for (int i = 0; i < x.length; i++) out[i] = Math.max(0.0, x[i]) / max;
        return out;
    }
    private static double[] clampNonNeg(double[] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) out[i] = Math.max(0.0, x[i]);
        return out;
    }

    public double[] getImportance()   { return importance.clone(); }
    public double[] getMIScores()     { return miScores.clone(); }
    public double[] getKSStatistics() { return ksStatistics.clone(); }

    public double[] projectToReduced(int[] selection) {
        if (selection == null) throw new IllegalArgumentException("selection must not be null");
        double[] reduced = new double[selection.length];
        for (int i = 0; i < selection.length; i++) {
            int origIdx = selection[i];
            if (origIdx >= 0 && origIdx < numFeatures) reduced[i] = importance[origIdx];
            else { reduced[i] = 0.0; projectionOutOfRange++; }
        }
        return reduced;
    }
}