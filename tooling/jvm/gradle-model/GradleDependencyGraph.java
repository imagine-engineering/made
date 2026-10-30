package dev.made.groovy;

public interface GradleDependencyGraph {
    String getJson();

    interface Parameters {
        String getConfiguration();
        void setConfiguration(String configuration);
    }
}
