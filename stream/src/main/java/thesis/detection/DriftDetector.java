package thesis.detection;

public interface DriftDetector {

    void update(double value);

    boolean isChangeDetected();

    default boolean isWarningDetected() { return false; }

    double getEstimation();

    void reset();

    default long getInputCount()  { return -1L; }

    default long getWindowLength() { return -1L; }

    default String name() { return getClass().getSimpleName(); }
}