package thesis.evaluation;

public class RecoveryTime {

    private final double tolerance;
    private final int maxRecoveryWindow;

    private boolean tracking;
    private double preDriftKappa;
    private long driftInstance;
    private long currentInstance;

    private int lastRecoveryTime = -1;
    private long totalRecoveryInstances;
    private int recoveredCount;
    private int unrecoveredCount;
    private int cancelledCount;
    private int driftAlarmCount;

    public RecoveryTime() { this(0.05, 10000); }

    public RecoveryTime(double tolerance, int maxRecoveryWindow) {
        if (!(tolerance >= 0 && tolerance <= 1)) throw new IllegalArgumentException("tolerance must be in [0,1]");
        if (maxRecoveryWindow < 1) throw new IllegalArgumentException("maxRecoveryWindow must be >= 1");
        this.tolerance = tolerance;
        this.maxRecoveryWindow = maxRecoveryWindow;
    }

    public void tick() { currentInstance++; }

    public void onDriftAlarm(double currentKappa) {
        driftAlarmCount++;
        if (!Double.isFinite(currentKappa)) currentKappa = 0.0;
        if (tracking) cancelledCount++;
        tracking = true;
        preDriftKappa = currentKappa;
        driftInstance = currentInstance;
    }

    public void update(double kappa) {
        if (!tracking) return;
        if (!Double.isFinite(kappa)) return;
        long elapsed = currentInstance - driftInstance;
        if (elapsed < 1) return;
        if (kappa >= preDriftKappa - tolerance) {
            lastRecoveryTime = (int) elapsed;
            totalRecoveryInstances += elapsed;
            recoveredCount++;
            tracking = false;
        } else if (elapsed >= maxRecoveryWindow) {
            unrecoveredCount++;
            tracking = false;
        }
    }

    public int getLastRecoveryTime()        { return lastRecoveryTime; }
    public double getAverageRecoveryTime()  { return recoveredCount == 0 ? Double.NaN : (double) totalRecoveryInstances / recoveredCount; }
    public int getRecoveredCount()          { return recoveredCount; }
    public int getUnrecoveredCount()        { return unrecoveredCount; }
    public int getCancelledCount()          { return cancelledCount; }
    public int getDriftAlarmCount()         { return driftAlarmCount; }
    public boolean isTracking()             { return tracking; }
    public double getPreDriftKappa()        { return preDriftKappa; }
    public long getCurrentInstance()        { return currentInstance; }

    public void reset() {
        tracking = false;
        preDriftKappa = 0.0;
        driftInstance = 0;
        currentInstance = 0;
        lastRecoveryTime = -1;
        totalRecoveryInstances = 0;
        recoveredCount = 0;
        unrecoveredCount = 0;
        cancelledCount = 0;
        driftAlarmCount = 0;
    }
}