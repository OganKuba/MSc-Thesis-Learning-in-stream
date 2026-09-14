package thesis.models;

import moa.MOAObject;
import moa.core.SizeOf;

public final class ModelSize {

    public static final long UNAVAILABLE = -1L;

    private ModelSize() {}

    public static boolean agentAvailable() {
        try {
            return SizeOf.sizeOf(new Object()) >= 0L;
        } catch (Throwable t) {
            return false;
        }
    }

    public static long of(MOAObject o) {
        if (o == null) return 0L;
        try {
            long b = o.measureByteSize();
            return b < 0L ? UNAVAILABLE : b;
        } catch (Throwable t) {
            return UNAVAILABLE;
        }
    }

    public static long sum(long... parts) {
        long total = 0L;
        for (long p : parts) {
            if (p < 0L) return UNAVAILABLE;
            total += p;
        }
        return total;
    }
}
