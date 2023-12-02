#!/bin/sh
# Makes a copy of the same wheel file but renames it so that it is published only
# for the supported OS, which is linux and macos.

dists=( "manylinux2014_x86_64" "manylinux2014_i686" "manylinux2014_aarch64" "manylinux2014_armv7l" "manylinux2014_ppc64" "manylinux2014_ppc64le" "manylinux2014_s390x" "macosx_10_9_x86_64" "macosx_11_0_arm64" )
for dist in "${dists[@]}"
do
    python -m build
    og_whl=$(ls dist/*-any.whl | sort | tail -n1)
    dist_whl=$(echo "$og_whl" | sed "s/any.whl/$dist.whl/g")
    mv $og_whl $dist_whl
done