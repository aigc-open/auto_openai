curl -X POST "http://127.0.0.1:9000/openai/v1/embeddings" \
-H "Authorization: Bearer xxxx" \
-H "Content-Type: application/json" \
-d '{
  "model": "bge-base-zh-v1.5",
  "input": [
    "Our solar system orbits the Milky Way galaxy at about 515,000 mph",
    "Jupiter'\''s Great Red Spot is a storm that has been raging for at least 350 years."
  ]
}'
