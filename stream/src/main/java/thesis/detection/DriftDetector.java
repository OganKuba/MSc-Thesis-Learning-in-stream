package thesis.detection;

public interface DriftDetector {

    void update(double value);

    boolean isChangeDetected();

    boolean isWarningDetected();

    double getEstimation();

    void reset();

    default String name() { return getClass().getSimpleName(); }
}