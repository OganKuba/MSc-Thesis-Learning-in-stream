package thesis.detection;

import lombok.Getter;

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
        public double kswinAlpha = 0.005;
        public int kswinWindowSize = 200;
        public double bhQ = 0.10;
        public int postDriftCooldown = 0;
        public boolean promoteReferenceOnDrift = true;

        public Config(int numFeatures) {
            this.numFeatures = numFeatures;
        }
    }

    private enum Phase { COLLECT_REF, MONITOR, COLLECT_POST }

    @Getter private final Config config;
    private final DriftDetector level1;
    private final FeatureBuffers buffers;
    private final PerFeatureKSWIN level2;

    private Phase phase;
    private int postCount;
    private int cooldownLeft;

    @Getter private boolean lastGlobalDrift;
    @Getter private boolean lastGlobalWarning;
    @Getter private Set<Integer> lastDriftingFeatures = Collections.emptySet();
    @Getter private double[] lastPValues;
    @Getter private long updateCount;
    @Getter private long globalAlarms;
    @Getter private long localizedAlarms;

    public TwoLevelDriftDetector(Config cfg) {
        if (cfg == null) throw new IllegalArgumentException("cfg must not be null");
        if (cfg.numFeatures < 1) throw new IllegalArgumentException("numFeatures must be >= 1");
        if (!(cfg.kswinAlpha > 0.0 && cfg.kswinAlpha < 1.0))
            throw new IllegalArgumentException("kswinAlpha must be in (0,1)");
        if (!(cfg.bhQ > 0.0 && cfg.bhQ < 1.0))
            throw new IllegalArgumentException("bhQ must be in (0,1)");
        if (cfg.kswinWindowSize < 10)
            throw new IllegalArgumentException("kswinWindowSize must be >= 10");
        if (cfg.postDriftCooldown < 0)
            throw new IllegalArgumentException("postDriftCooldown must be >= 0");

        this.config = cfg;
        this.level1 = buildLevel1(cfg);
        this.buffers = new FeatureBuffers(cfg.numFeatures, cfg.kswinWindowSize);
        this.level2 = new PerFeatureKSWIN(cfg.numFeatures, cfg.kswinAlpha,
                cfg.kswinWindowSize, cfg.bhQ);
        this.lastPValues = new double[cfg.numFeatures];
        java.util.Arrays.fill(this.lastPValues, 1.0);
        this.phase = Phase.COLLECT_REF;
        this.postCount = 0;
        this.cooldownLeft = 0;
    }

    private static DriftDetector buildLevel1(Config cfg) {
        switch (cfg.level1Type) {
            case ADWIN:  return new ADWINChangeDetector(cfg.level1Delta);
            case HDDM_A: return HDDMChangeDetector.ofA(cfg.level1Delta, cfg.level1AlphaW);
            case HDDM_W: return HDDMChangeDetector.ofW(cfg.level1Delta, cfg.level1AlphaW, cfg.level1Lambda);
            default: throw new IllegalStateException("unknown Level1Type: " + cfg.level1Type);
        }
    }

    public void update(double predictionError, double[] featureValues) {
        if (featureValues == null || featureValues.length != config.numFeatures) {
            throw new IllegalArgumentException(
                    "expected " + config.numFeatures + " features, got " +
                            (featureValues == null ? "null" : featureValues.length));
        }
        updateCount++;

        buffers.pushRolling(featureValues);

        lastGlobalDrift = false;
        lastGlobalWarning = false;
        lastDriftingFeatures = Collections.emptySet();

        if (cooldownLeft > 0) {
            cooldownLeft--;
            level1.update(predictionError);
            return;
        }

        if (phase == Phase.COLLECT_POST) {
            buffers.pushPost(featureValues);
            postCount++;
            if (postCount >= config.kswinWindowSize) {
                runLevel2Localization();
                phase = Phase.MONITOR;
                postCount = 0;
                cooldownLeft = config.postDriftCooldown;
                if (lastDriftingFeatures != null && !lastDriftingFeatures.isEmpty()) {
                    lastGlobalDrift = true;
                    globalAlarms++;
                }
            }
            level1.update(predictionError);
            return;
        }


        level1.update(predictionError);
        lastGlobalWarning = level1.isWarningDetected();

        if (level1.isChangeDetected()) {
            globalAlarms++;
            lastGlobalDrift = true;
            buffers.snapshotReferenceFromRolling();
            buffers.clearPost();
            postCount = 0;
            phase = Phase.COLLECT_POST;
            level1.reset();
        } else if (phase == Phase.COLLECT_REF && buffers.isRollingFull()) {
            phase = Phase.MONITOR;
        }
    }

    private void runLevel2Localization() {
        double[][] ref = buffers.getReference();
        double[][] post = buffers.getPost();
        if (ref == null) return;
        Set<Integer> drifting = level2.testWindows(ref, post);
        double[] p = level2.getLastPValues();
        System.arraycopy(p, 0, lastPValues, 0, p.length);
        lastDriftingFeatures = drifting;
        if (!drifting.isEmpty()) localizedAlarms++;
        if (config.promoteReferenceOnDrift) {
            buffers.promotePostToReference();
        }
    }

    public boolean isGlobalDriftDetected()   { return lastGlobalDrift; }
    public boolean isGlobalWarningDetected() { return lastGlobalWarning; }
    public Set<Integer> getDriftingFeatureIndices() { return lastDriftingFeatures; }
    public double getLevel1Estimation() { return level1.getEstimation(); }
    public String level1Name() { return level1.name(); }

    public void reset() {
        level1.reset();
        buffers.reset();
        java.util.Arrays.fill(lastPValues, 1.0);
        lastGlobalDrift = false;
        lastGlobalWarning = false;
        lastDriftingFeatures = Collections.emptySet();
        updateCount = 0;
        globalAlarms = 0;
        localizedAlarms = 0;
        phase = Phase.COLLECT_REF;
        postCount = 0;
        cooldownLeft = 0;
    }
}