package thesis.detection;

import lombok.Getter;
import moa.classifiers.core.driftdetection.AbstractChangeDetector;
import moa.classifiers.core.driftdetection.HDDM_A_Test;
import moa.classifiers.core.driftdetection.HDDM_W_Test;

import java.util.Locale;
import java.util.OptionalDouble;

public class HDDMChangeDetector implements DriftDetector {

    public enum Variant { A, W }

    @Getter private final Variant variant;
    @Getter private final double alphaD;
    @Getter private final double alphaW;
    private final double lambda;

    private AbstractChangeDetector detector;
    private long inputs;
    private long drifts;

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

    public OptionalDouble getLambda() {
        return variant == Variant.W ? OptionalDouble.of(lambda) : OptionalDouble.empty();
    }

    public long getInputCount() { return inputs; }
    public long getDriftCount() { return drifts; }

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
        if (!Double.isFinite(value)) return;
        detector.input(value);
        inputs++;
        if (detector.getChange()) drifts++;
    }

    @Override public boolean isChangeDetected()  { return detector.getChange(); }
    @Override public boolean isWarningDetected() { return detector.getWarningZone(); }
    @Override public double  getEstimation()     { return detector.getEstimation(); }

    @Override
    public void reset() {
        this.detector = build();
    }

    @Override
    public String name() {
        if (variant == Variant.A) {
            return String.format(Locale.ROOT,
                    "HDDM_A(alphaD=%.6f, alphaW=%.6f)", alphaD, alphaW);
        }
        return String.format(Locale.ROOT,
                "HDDM_W(alphaD=%.6f, alphaW=%.6f, lambda=%.6f)", alphaD, alphaW, lambda);
    }
}