package thesis.models;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import lombok.Getter;

@Getter
public final class FeatureSpace {

    private final InstancesHeader header;
    private final int classIdx;
    private final int[] featureToAttr;

    public FeatureSpace(InstancesHeader header) {
        if (header == null) throw new IllegalArgumentException("header must not be null");
        if (header.classIndex() < 0) {
            throw new IllegalArgumentException("header has no class attribute");
        }
        this.header = header;
        this.classIdx = header.classIndex();
        int d = header.numAttributes() - 1;
        this.featureToAttr = new int[d];
        int j = 0;
        for (int i = 0; i < header.numAttributes(); i++) {
            if (i == classIdx) continue;
            featureToAttr[j++] = i;
        }
    }

    public int numFeatures()               { return featureToAttr.length; }
    public int classIndex()                { return classIdx; }
    public int attrIndexOf(int featureIdx) {
        if (featureIdx < 0 || featureIdx >= featureToAttr.length) {
            throw new IndexOutOfBoundsException("featureIdx=" + featureIdx);
        }
        return featureToAttr[featureIdx];
    }

    public double[] extractFeatures(Instance full) {
        double[] out = new double[featureToAttr.length];
        for (int i = 0; i < featureToAttr.length; i++) {
            out[i] = full.value(featureToAttr[i]);
        }
        return out;
    }
}