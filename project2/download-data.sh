#!/usr/bin/env bash
# download_and_unpack.sh
# ----------------------
# Downloads ALL Middlebury 2014 stereo zip archives into
#   ./data/middlebury-stereo/zip
# then unpacks each into
#   ./data/middlebury-stereo/

set -e

BASE_URL="https://vision.middlebury.edu/stereo/data/scenes2014/zip"
DEST_ROOT="$HOME/data/middlebury-stereo-2014"
ZIP_ROOT="$DEST_ROOT/zip"

mkdir -p "$DEST_ROOT"

echo "▶ Downloading all .zip archives into '$ZIP_ROOT'."

echo "▶ Fetching list of ZIP filenames..."
zip_files=$(wget -qO- "$BASE_URL/" \
  | grep -oE 'href="[^"]+\.zip"' \
  | sed -E 's/.*href="([^"]+)".*/\1/')

count=$(echo $zip_files | wc -w | xargs)

echo "▶ Downloading $count archives in parallel..."
printf "%s\n" $zip_files | xargs -n1 -P8 -I{} wget -nc -P $ZIP_ROOT -q "$BASE_URL/{}"

echo "▶ Unzipping all archives into $DEST_ROOT..."
cd $ZIP_ROOT
for z in *.zip; do
  echo "   • Unpacking $z"
  unzip -o "$z" -d "../" > /dev/null
done

echo "✅ Done!"
