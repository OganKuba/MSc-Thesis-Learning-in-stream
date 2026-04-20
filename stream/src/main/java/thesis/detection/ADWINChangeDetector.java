package thesis.detection;

import moa.classifiers.core.driftdetection.ADWIN;

public class ADWINChangeDetector implements DriftDetector {

    private final double delta;
    private ADWIN adwin;
    private boolean changeDetected;

    public ADWINChangeDetector() {
        this(0.002);
    }

    public ADWINChangeDetector(double delta) {
        if (delta <= 0.0 || delta >= 1.0) {
            throw new IllegalArgumentException("delta must be in (0,1), got " + delta);
        }
        this.delta = delta;
        this.adwin = new ADWIN(delta);
        this.changeDetected = false;
    }

    @Override
    public void update(double value) {
        this.changeDetected = this.adwin.setInput(value);
    }

    @Override
    public boolean isChangeDetected() {
        return changeDetected;
    }

    @Override
    public boolean isWarningDetected() {
        return false;
    }

    @Override
    public double getEstimation() {
        return adwin.getEstimation();
    }

    @Override
    public void reset() {
        this.adwin = new ADWIN(delta);
        this.changeDetected = false;
    }

    public double getDelta() {
        return delta;
    }

    public int getWindowLength() {
        return (int) adwin.getWidth();
    }

    @Override
    public String name() {
        return "ADWIN(delta=" + delta + ")";
    }
}