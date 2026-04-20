package thesis.detection;

import moa.classifiers.core.driftdetection.AbstractChangeDetector;
import moa.classifiers.core.driftdetection.HDDM_A_Test;
import moa.classifiers.core.driftdetection.HDDM_W_Test;

public class HDDMChangeDetector implements DriftDetector {

    public enum Variant { A, W }

    private final Variant variant;
    private final double alphaD;
    private final double alphaW;
    private final double lambda;
    private AbstractChangeDetector detector;

    public static HDDMChangeDetector ofA(double alphaD, double alphaW) {
        return new HDDMChangeDetector(Variant.A, alphaD, alphaW, Double.NaN);
    }

    public static HDDMChangeDetector ofW(double alphaD, double alphaW, double lambda) {
        return new HDDMChangeDetector(Variant.W, alphaD, alphaW, lambda);
    }

    public HDDMChangeDetector(Variant variant, double alphaD, double alphaW, double lambda) {
        if (variant == null) {
            throw new IllegalArgumentException("variant must not be null");
        }
        if (!(alphaD > 0.0 && alphaD < 1.0)) {
            throw new IllegalArgumentException("alphaD must be in (0,1), got " + alphaD);
        }
        if (!(alphaW > 0.0 && alphaW < 1.0)) {
            throw new IllegalArgumentException("alphaW must be in (0,1), got " + alphaW);
        }
        if (alphaW < alphaD) {
            throw new IllegalArgumentException(
                    "alphaW must be >= alphaD (warning fires earlier than drift)");
        }
        if (variant == Variant.W && !(lambda > 0.0 && lambda < 1.0)) {
            throw new IllegalArgumentException("lambda must be in (0,1) for HDDM_W, got " + lambda);
        }
        this.variant = variant;
        this.alphaD = alphaD;
        this.alphaW = alphaW;
        this.lambda = lambda;
        this.detector = build();
    }

    private AbstractChangeDetector build() {
        if (variant == Variant.A) {
            HDDM_A_Test d = new HDDM_A_Test();
            d.driftConfidenceOption.setValue(alphaD);
            d.warningConfidenceOption.setValue(alphaW);
            d.prepareForUse();
            return d;
        } else {
            HDDM_W_Test d = new HDDM_W_Test();
            d.driftConfidenceOption.setValue(alphaD);
            d.warningConfidenceOption.setValue(alphaW);
            d.lambdaOption.setValue(lambda);
            d.prepareForUse();
            return d;
        }
    }

    @Override
    public void update(double value) {
        detector.input(value);
    }

    @Override
    public boolean isChangeDetected() {
        return detector.getChange();
    }

    @Override
    public boolean isWarningDetected() {
        return detector.getWarningZone();
    }

    @Override
    public double getEstimation() {
        return detector.getEstimation();
    }

    @Override
    public void reset() {
        this.detector = build();
    }

    public Variant getVariant() { return variant; }
    public double getAlphaD()   { return alphaD; }
    public double getAlphaW()   { return alphaW; }
    public double getLambda()   { return lambda; }

    @Override
    public String name() {
        if (variant == Variant.A) {
            return "HDDM_A(alphaD=" + alphaD + ", alphaW=" + alphaW + ")";
        }
        return "HDDM_W(alphaD=" + alphaD + ", alphaW=" + alphaW + ", lambda=" + lambda + ")";
    }
}