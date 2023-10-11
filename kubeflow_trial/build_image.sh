#!/bin/bash -e
image_name=gcr.io/moth-recognition/simple_component_trial
image_tag=latest
full_image_name=${image_name}:${image_tag}

cd "$(dirname "$0")" 
docker buildx build --platform linux/amd64 -f ./Dockerfile -t "${full_image_name}" .
docker push "$full_image_name"
#
## Output the strict image name, which contains the sha256 image digest
docker inspect --format="{{index .RepoDigests 0}}" "${full_image_name}"
