#!/usr/bin/env bash
set -euo pipefail

STREAM_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$STREAM_DIR"

JAVA_HOME="${JAVA_HOME:-/home/kubog/.jdks/temurin-17.0.19}"
JAVAC="$JAVA_HOME/bin/javac"
JAVA="$JAVA_HOME/bin/java"
M2="$HOME/.m2/repository"

CP="$M2/nz/ac/waikato/cms/moa/moa/2024.07.0/moa-2024.07.0.jar"
CP="$CP:$M2/org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1.jar"
CP="$CP:$M2/com/fasterxml/jackson/core/jackson-databind/2.17.2/jackson-databind-2.17.2.jar"
CP="$CP:$M2/com/fasterxml/jackson/core/jackson-core/2.17.2/jackson-core-2.17.2.jar"
CP="$CP:$M2/com/fasterxml/jackson/core/jackson-annotations/2.17.0/jackson-annotations-2.17.0.jar"
LOMBOK="$M2/org/projectlombok/lombok/1.18.30/lombok-1.18.30.jar"
SIZEOF_AGENT="$M2/com/github/fracpete/sizeofag/1.1.0/sizeofag-1.1.0.jar"
if [[ ! -f "$SIZEOF_AGENT" ]]; then
  echo "[run] ERROR: sizeof agent not found at $SIZEOF_AGENT" >&2
  echo "[run]        RAM-Hours would be NaN for every run. Fetch it with:" >&2
  echo "[run]        mvn dependency:get -Dartifact=com.github.fracpete:sizeofag:1.1.0" >&2
  exit 1
fi

CONFIG="${1:-src/main/java/thesis/experiments/master_experiments.json}"
BUILD_DIR="target/classes"

echo "[run] JDK: $($JAVA -version 2>&1 | head -1)"
echo "[run] compiling to $BUILD_DIR ..."
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"
find src/main/java -name '*.java' > /tmp/srcs_run.txt
"$JAVAC" -d "$BUILD_DIR" -cp "$CP:$LOMBOK" -processorpath "$LOMBOK" @/tmp/srcs_run.txt

echo "[run] launching experiments with config: $CONFIG"
echo "[run] NOTE: this overwrites stream/results/ — back it up first if you need the old run."
"$JAVA" -javaagent:"$SIZEOF_AGENT" -cp "$BUILD_DIR:$CP" \
    thesis.experiments.UnifiedStreamExperimentRunner "$CONFIG"

echo "[run] done. Regenerate figures/tables with:  analysis/.venv/bin/python -m analysis"
