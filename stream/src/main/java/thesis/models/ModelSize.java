package thesis.models;

import moa.MOAObject;
import moa.core.SizeOf;

/**
 * Deep model-size measurement for the RAM-Hours metric.
 *
 * <p>RAM-Hours (Bifet et al., 2010) is defined as <i>gigabytes of memory held by the
 * <b>model</b>, integrated over time</i> — it is a property of the learner, not of the JVM the
 * learner happens to run in. Sampling {@code Runtime.totalMemory() - freeMemory()} instead (the
 * previous implementation) measures the whole heap, which under the runner's fixed-size thread
 * pool is shared by up to {@code num_threads} concurrent runs. That made the recorded figure a
 * function of how many neighbours happened to be running, not of the model: e.g. a
 * {@link MajorityClassWrapper} — which allocates one {@code long[numClasses]} — was credited with
 * 3.5 GB on NHTS and 164 MB on SEA in the same batch.
 *
 * <p>MOA's {@link MOAObject#measureByteSize()} walks the object graph through the {@code sizeofag}
 * java agent. Without {@code -javaagent:sizeofag-<version>.jar} on the command line every call
 * returns {@code -1}; we propagate that as "unavailable" rather than substituting a plausible-looking
 * number, so a misconfigured run yields {@code NaN} instead of silently fabricated memory figures.
 *
 * @see thesis.evaluation.RAMHours
 */
public final class ModelSize {

    /** Sentinel for "the sizeof agent is not loaded, so no measurement is possible". */
    public static final long UNAVAILABLE = -1L;

    private ModelSize() {}

    /**
     * @return {@code true} when the {@code sizeofag} java agent is loaded and deep measurement works.
     */
    public static boolean agentAvailable() {
        try {
            return SizeOf.sizeOf(new Object()) >= 0L;
        } catch (Throwable t) {
            return false;
        }
    }

    /** Deep size of one MOA object, or {@link #UNAVAILABLE}. */
    public static long of(MOAObject o) {
        if (o == null) return 0L;
        try {
            long b = o.measureByteSize();
            return b < 0L ? UNAVAILABLE : b;
        } catch (Throwable t) {
            return UNAVAILABLE;
        }
    }

    /**
     * Sum of component sizes, propagating {@link #UNAVAILABLE}: if any component could not be
     * measured the total is unavailable too (a partial sum would understate the model).
     */
    public static long sum(long... parts) {
        long total = 0L;
        for (long p : parts) {
            if (p < 0L) return UNAVAILABLE;
            total += p;
        }
        return total;
    }
}
