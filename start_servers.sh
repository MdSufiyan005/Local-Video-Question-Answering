#!/usr/bin/env bash
cd "$(dirname "$0")"
export PATH="$PWD/llama.cpp/build/bin:$PATH"
pkill llama-server 2>/dev/null; sleep 1

VLM=$(find models -name "*Q4_K_M*.gguf" ! -name "*mmproj*" | head -1)
MMPROJ=$(find models -name "*mmproj*.gguf" | head -1)
EMB=$(find models -ipath "*embed*" -name "*.gguf" | head -1)
echo "VLM=$VLM"; echo "MMPROJ=$MMPROJ"; echo "EMB=$EMB"
for f in "$VLM" "$MMPROJ" "$EMB"; do
  [ -f "$f" ] || { echo "ERROR: a model file is missing (see the paths above). Re-run the hf download commands."; exit 1; }
done

llama-server -m "$VLM" --mmproj "$MMPROJ" --host 127.0.0.1 --port 8080 -ngl 99 -c 8192 -np 1 > vlm.log 2>&1 &
llama-server -m "$EMB" --host 127.0.0.1 --port 8081 -ngl 99 --embedding --pooling mean -c 512 -b 512 -ub 512 -np 1 > embed.log 2>&1 &
echo "Started. Check: tail -f vlm.log embed.log"
