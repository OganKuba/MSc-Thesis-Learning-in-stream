package thesis.models;

import java.util.Arrays;

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

    public FeatureImportance(int numFeatures) {
        this(numFeatures, 0.7, 0.3, 1e-6, true);
    }

    public FeatureImportance(int numFeatures, double w1, double w2,
                             double epsilon, boolean normalizeInputs) {
        if (numFeatures < 1) throw new IllegalArgumentException("numFeatures must be >= 1");
        if (w1 < 0.0 || w2 < 0.0) {
            throw new IllegalArgumentException("weights must be non-negative");
        }
        if (w1 + w2 == 0.0) {
            throw new IllegalArgumentException("at least one of w1, w2 must be > 0");
        }
        if (epsilon <= 0.0) throw new IllegalArgumentException("epsilon must be > 0");
        this.numFeatures = numFeatures;
        this.w1 = w1;
        this.w2 = w2;
        this.epsilon = epsilon;
        this.normalizeInputs = normalizeInputs;
        this.miScores = new double[numFeatures];
        this.ksStatistics = new double[numFeatures];
        this.importance = new double[numFeatures];
        Arrays.fill(this.importance, 1.0 / numFeatures);
    }

    public void update(double[] miScores, double[] ksStatistics) {
        if (miScores == null || miScores.length != numFeatures) {
            throw new IllegalArgumentException("miScores must have length " + numFeatures);
        }
        if (ksStatistics == null || ksStatistics.length != numFeatures) {
            throw new IllegalArgumentException("ksStatistics must have length " + numFeatures);
        }

        double[] mi = normalizeInputs ? maxNormalize(miScores) : miScores.clone();
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

        if (sumImp <= 0.0) {
            Arrays.fill(imp, 1.0 / numFeatures);
        } else {
            for (int i = 0; i < numFeatures; i++) imp[i] /= sumImp;
        }

        this.miScores = miScores.clone();
        this.ksStatistics = ksStatistics.clone();
        this.importance = imp;
        this.updates++;
    }

    public void updateMIOnly(double[] miScores) {
        update(miScores, this.ksStatistics);
    }

    public void updateKSOnly(double[] ksStatistics) {
        update(this.miScores, ksStatistics);
    }

    private static double[] maxNormalize(double[] x) {
        double max = 0.0;
        for (double v : x) if (v > max) max = v;
        if (max <= 0.0) return x.clone();
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) out[i] = Math.max(0.0, x[i]) / max;
        return out;
    }

    public double[] getImportance()    { return importance.clone(); }
    public double[] getMIScores()      { return miScores.clone(); }
    public double[] getKSStatistics()  { return ksStatistics.clone(); }
    public int getNumFeatures()        { return numFeatures; }
    public double getW1()              { return w1; }
    public double getW2()              { return w2; }
    public long getUpdates()           { return updates; }

    public double[] projectToReduced(int[] selection) {
        double[] reduced = new double[selection.length];
        for (int i = 0; i < selection.length; i++) {
            int origIdx = selection[i];
            reduced[i] = (origIdx >= 0 && origIdx < numFeatures) ? importance[origIdx] : 0.0;
        }
        return reduced;
    }
}