#!/bin/bash

die() {
  echo >&2 "$*"
  exit 1
}

dest=$1
[ "$dest" ] || die "usage: $0 jevois-mountpoint"
[ -d "$dest" ] || die "invalid target directory"

copy_file() {
  local output="$1/$2"
  mkdir -p "$(dirname "$output")"
  cp -v "$2" "$output"
}

echo "copying source files"
(
  cd src/Modules
  for f in $(git ls-files); do
    copy_file "$dest/modules/RobOtter" "$f"
  done
)

echo "copying rome files"
(
  cd robotter
  for f in $(git ls-files rome); do
    copy_file "$dest/modules/RobOtter/CubeVision" "$f"
  done
)

echo "copying rome_messages.py"
cp -v "eurobot/rome_messages.py" "$dest/modules/RobOtter/CubeVision"

find "$dest" -name __pycache__ -exec rm -vfr {} \;

