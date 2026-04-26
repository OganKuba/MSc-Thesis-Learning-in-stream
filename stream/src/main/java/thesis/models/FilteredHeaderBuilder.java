package thesis.models;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import java.util.ArrayList;

public final class FilteredHeaderBuilder {

    private FilteredHeaderBuilder() {}

    public static InstancesHeader build(FeatureSpace space, int[] selection, String suffix) {
        InstancesHeader full = space.getHeader();
        ArrayList<Attribute> attrs = new ArrayList<>(selection.length + 1);
        for (int idx : selection) {
            attrs.add(full.attribute(space.attrIndexOf(idx)));
        }
        attrs.add(full.classAttribute());
        Instances ins = new Instances(full.getRelationName() + suffix, attrs, 0);
        ins.setClassIndex(attrs.size() - 1);
        return new InstancesHeader(ins);
    }

    public static Instance filteredInstance(Instance full, FeatureSpace space,
                                            int[] selection,
                                            InstancesHeader filteredHeader) {
        double[] vals = new double[selection.length + 1];
        for (int i = 0; i < selection.length; i++) {
            vals[i] = full.value(space.attrIndexOf(selection[i]));
        }
        if (!full.classIsMissing()) {
            vals[selection.length] = full.classValue();
        }
        Instance out = new DenseInstance(full.weight(), vals);
        out.setDataset(filteredHeader);
        return out;
    }
}