# Common environment setup across build*.sh scripts

export VERSION=${VERSION:-$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$VERSION\" \"-X=github.com/ollama/ollama/server.mode=release\"'"
DOCKER_ORG=${DOCKER_ORG:-"ollama"}
RELEASE_IMAGE_REPO=${RELEASE_IMAGE_REPO:-"${DOCKER_ORG}/release"}
FINAL_IMAGE_REPO=${FINAL_IMAGE_REPO:-"${DOCKER_ORG}/ollama"}
OLLAMA_COMMON_BUILD_ARGS="--build-arg=VERSION --build-arg=GOFLAGS --build-arg=OLLAMA_CUSTOM_CPU_DEFS --build-arg=AMDGPU_TARGETS"

echo "Building Ollama version $VERSION"