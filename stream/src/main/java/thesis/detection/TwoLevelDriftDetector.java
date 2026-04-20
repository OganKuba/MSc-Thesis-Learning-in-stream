package thesis.detection;

import java.util.Collections;
import java.util.Set;

public class TwoLevelDriftDetector {

    public enum Level1Type { ADWIN, HDDM_A, HDDM_W }

    public static final class Config {
        public Level1Type level1Type = Level1Type.ADWIN;
        public double level1Delta = 0.002;
        public double level1AlphaW = 0.01;
        public double level1Lambda = 0.05;
        public int numFeatures;
        public double kswinAlpha = 0.01;
        public int kswinWindowSize = 300;
        public double bhQ = 0.05;
        public boolean promoteReferenceOnDrift = true;

        public Config(int numFeatures) {
            this.numFeatures = numFeatures;
        }
    }

    private final Config cfg;
    private final DriftDetector level1;
    private final PerFeatureKSWIN level2;

    private boolean lastGlobalDrift;
    private boolean lastGlobalWarning;
    private Set<Integer> lastDriftingFeatures;
    private long updates;
    private long globalAlarms;

    public TwoLevelDriftDetector(Config cfg) {
        if (cfg == null) {
            throw new IllegalArgumentException("cfg must not be null");
        }
        if (cfg.numFeatures < 1) {
            throw new IllegalArgumentException("numFeatures must be >= 1");
        }
        this.cfg = cfg;
        this.level1 = buildLevel1(cfg);
        this.level2 = new PerFeatureKSWIN(cfg.numFeatures, cfg.kswinAlpha,
                cfg.kswinWindowSize, cfg.bhQ);
        this.lastGlobalDrift = false;
        this.lastGlobalWarning = false;
        this.lastDriftingFeatures = Collections.emptySet();
        this.updates = 0;
        this.globalAlarms = 0;
    }

    private static DriftDetector buildLevel1(Config cfg) {
        switch (cfg.level1Type) {
            case ADWIN:  return new ADWINChangeDetector(cfg.level1Delta);
            case HDDM_A: return HDDMChangeDetector.ofA(cfg.level1Delta, cfg.level1AlphaW);
            case HDDM_W: return HDDMChangeDetector.ofW(cfg.level1Delta, cfg.level1AlphaW,
                    cfg.level1Lambda);
            default: throw new IllegalStateException("unknown Level1Type: " + cfg.level1Type);
        }
    }

    public void update(double predictionError, double[] featureValues) {
        if (featureValues == null || featureValues.length != cfg.numFeatures) {
            throw new IllegalArgumentException(
                    "expected " + cfg.numFeatures + " features, got " +
                            (featureValues == null ? "null" : featureValues.length));
        }
        updates++;

        level2.update(featureValues);
        level1.update(predictionError);

        lastGlobalWarning = level1.isWarningDetected();
        lastGlobalDrift = level1.isChangeDetected();

        if (lastGlobalDrift) {
            globalAlarms++;
            lastDriftingFeatures = level2.getDriftingFeatures();
            if (cfg.promoteReferenceOnDrift && level2.isReady()) {
                for (int idx : lastDriftingFeatures) {
                    level2.resetFeature(idx);
                }
            }
        } else {
            lastDriftingFeatures = Collections.emptySet();
        }
    }

    public boolean isGlobalDriftDetected()   { return lastGlobalDrift; }
    public boolean isGlobalWarningDetected() { return lastGlobalWarning; }

    public Set<Integer> getDriftingFeatureIndices() {
        return lastDriftingFeatures;
    }

    public double[] getLastPValues() {
        return level2.getLastPValues();
    }

    public double getLevel1Estimation() {
        return level1.getEstimation();
    }

    public boolean isLevel2Ready() {
        return level2.isReady();
    }

    public long getUpdateCount()  { return updates; }
    public long getGlobalAlarms() { return globalAlarms; }

    public Config getConfig()     { return cfg; }
    public String level1Name()    { return level1.name(); }

    public void reset() {
        level1.reset();
        level2.resetAll();
        lastGlobalDrift = false;
        lastGlobalWarning = false;
        lastDriftingFeatures = Collections.emptySet();
        updates = 0;
        globalAlarms = 0;
    }
}