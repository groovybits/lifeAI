#!/bin/sh
#
for v in nova alloy echo fable onyx nova shimmer; do
    curl https://api.openai.com/v1/audio/speech \
      -H "Authorization: Bearer $OPENAI_API_KEY" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"tts-1\",
        \"input\": \"AI is amazing and Anime is good. It is a miracle that GPT-4 is so good.\",
        \"voice\": \"$v\",
        \"response_format\": \"aac\",
        \"speed\": \"1.0\"
      }" \
      --output speech_$v.aac
done
