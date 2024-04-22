#!/bin/bash

domains="../experiments/data/priv_domains.json"
script=secmi_sd_laion.sh

jq -r '.[]' "$domains" | while read name; do
    echo "MIA attacking domain: $name..."
    sh ./scripts/$script '$name'
done
