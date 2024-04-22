#!/bin/bash

domains="./data/priv_domains.json"
script=lora_db.sh

jq -r '.[]' "$domains" | while read name; do
    echo "Finetuning domain: $name..."
    sh ./scripts/$script '$name'
done
