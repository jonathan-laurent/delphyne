#!/bin/bash

# Requires imagemagick
# brew update && brew install imagemagick

LOGOS_DIR=$(dirname "$0")

for file in $LOGOS_DIR/raw/*.webp $LOGOS_DIR/raw/*.png; do
    new_file=$(basename -- "$file")
    new_file="${new_file%.*}.png"
    new_file="$LOGOS_DIR/converted/$new_file"
    echo "Converting: $file > $new_file"
    convert "$file" -fuzz 10% -transparent white -trim +repage "$new_file"
done