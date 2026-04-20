package thesis.pipeline;

import moa.streams.ArffFileStream;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import java.util.HashMap;
import java.util.Map;

public class ArffSanityCheck {

    public static void checkArff(String path) {
        System.out.println("=".repeat(60));
        System.out.println("FILE: " + path);
        System.out.println("=".repeat(60));

        ArffFileStream stream = new ArffFileStream(path, -1);
        stream.prepareForUse();

        InstancesHeader header = stream.getHeader();
        int numAttrs = header.numAttributes();
        int classIndex = header.classIndex();

        System.out.println("Attributes: " + numAttrs + " (class index: " + classIndex + ")");
        System.out.println("\nAttribute list:");
        for (int i = 0; i < numAttrs; i++) {
            String type = header.attribute(i).isNumeric() ? "NUMERIC" : "NOMINAL";
            String marker = (i == classIndex) ? " ← TARGET" : "";
            System.out.printf("  [%2d] %-30s %s%s%n", i, header.attribute(i).name(), type, marker);
        }

        Map<String, Integer> classCounts = new HashMap<>();
        int totalInstances = 0;

        System.out.println("\nFirst 5 instances:");
        while (stream.hasMoreInstances()) {
            Instance inst = stream.nextInstance().getData();
            totalInstances++;

            String classVal = header.attribute(classIndex).value((int) inst.classValue());
            classCounts.merge(classVal, 1, Integer::sum);

            if (totalInstances <= 5) {
                StringBuilder sb = new StringBuilder("  [" + totalInstances + "] ");
                for (int i = 0; i < inst.numAttributes(); i++) {
                    if (i == classIndex) {
                        sb.append(classVal);
                    } else {
                        sb.append(String.format("%.4f", inst.value(i)));
                    }
                    if (i < inst.numAttributes() - 1) sb.append(", ");
                }
                System.out.println(sb);
            }
        }

        System.out.println("\nTotal instances: " + totalInstances);
        System.out.println("Class distribution:");
        int finalTotal = totalInstances;
        classCounts.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .forEach(e -> System.out.printf("  %-20s %7d  (%.1f%%)%n",
                        e.getKey(), e.getValue(), 100.0 * e.getValue() / finalTotal));
        System.out.println();
    }

    public static void main(String[] args) {
        String base = "/home/kubog/MSc-Thesis-Learning-in-stream/preproccessing";

        checkArff(base + "/data/arff/nhts.arff");
        checkArff(base + "/nyc_taxi/data/arff/nyc_taxi.arff");
        checkArff(base + "/yahoo_finance/data/arff/yahoo_finance.arff");
    }
}