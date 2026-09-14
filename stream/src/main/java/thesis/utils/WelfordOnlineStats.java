package thesis.utils;

import lombok.Getter;

@Getter
public class WelfordOnlineStats {

    private long count;
    private double mean;
    private double m2;

    public WelfordOnlineStats() {
        reset();
    }

    public void update(double value) {
        count++;
        double delta = value - mean;
        mean += delta / count;
        double delta2 = value - mean;
        m2 += delta * delta2;
    }

    public double getMean() {
        return count == 0 ? 0.0 : mean;
    }

    public double getVariance() {
        if (count < 2) return 0.0;
        return m2 / (count - 1);
    }

    public double getStdDev() {
        return Math.sqrt(getVariance());
    }

    public void reset() {
        count = 0;
        mean = 0.0;
        m2 = 0.0;
    }
}