#!/bin/sh
# Makes a copy of the same wheel file but renames it so that it is published only
# for the supported OS, which is linux and macos.
python -m build
og_whl=$(ls dist/*-any.whl | sort | tail -n1)
linux_whl=$(echo "$og_whl" | sed "s/any.whl/manylinux1_x86_64.whl/g")
macos_10_whl=$(echo "$og_whl" | sed "s/any.whl/macosx_10_9_x86_64.whl/g")
macos_11_whl=$(echo "$og_whl" | sed "s/any.whl/macosx_11_0_arm64.whl/g")
cp $og_whl $linux_whl
cp $og_whl $macos_10_whl
cp $og_whl $macos_11_whl
rm $og_whl