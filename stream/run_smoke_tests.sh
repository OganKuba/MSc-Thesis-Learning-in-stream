#!/usr/bin/env bash
set -uo pipefail

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
# Some metric tests measure model size, which needs the sizeof agent.
AGENT="$M2/com/github/fracpete/sizeofag/1.1.0/sizeofag-1.1.0.jar"
AGENT_OPT=""
[[ -f "$AGENT" ]] && AGENT_OPT="-javaagent:$AGENT"

BUILD_DIR="target/smoke-classes"
echo "[smoke] compiling to $BUILD_DIR ..."
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"
find src/main/java -name '*.java' > "$BUILD_DIR/srcs.txt"
"$JAVAC" -d "$BUILD_DIR" -cp "$CP:$LOMBOK" -processorpath "$LOMBOK" \
    @"$BUILD_DIR/srcs.txt" 2>&1 | grep -v '^Note:' || true

total_pass=0; total_fail=0; bad=()
for f in $(find src/main/java -name '*SmokeTest.java' | sort); do
  cls=$(echo "$f" | sed 's|src/main/java/||; s|/|.|g; s|\.java||')
  out=$(timeout 600 "$JAVA" $AGENT_OPT -cp "$BUILD_DIR:$CP" "$cls" 2>&1 | grep -E '^RESULT' | head -1)
  p=$(grep -oE '[0-9]+ passed' <<<"$out" | grep -oE '[0-9]+' || echo 0)
  fl=$(grep -oE '[0-9]+ failed' <<<"$out" | grep -oE '[0-9]+' || echo 0)
  total_pass=$((total_pass + p)); total_fail=$((total_fail + fl))
  status="ok"
  if [[ "$fl" != "0" ]]; then status="FAILED"; bad+=("$cls"); fi
  printf '  %-56s %s\n' "${cls#thesis.}" "${out:-no RESULT line} [$status]"
done

echo "----------------------------------------------------------------"
echo "[smoke] TOTAL: $total_pass passed, $total_fail failed"
if (( total_fail > 0 )); then
  printf '[smoke] failing classes: %s\n' "${bad[*]}"
  exit 1
fi
