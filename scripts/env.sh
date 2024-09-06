# Common environment setup across build*.sh scripts

export VERSION=${VERSION:-$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$VERSION\" \"-X=github.com/ollama/ollama/server.mode=release\"'"
DOCKER_ORG=${DOCKER_ORG:-"ollama"}
RELEASE_IMAGE_REPO=${RELEASE_IMAGE_REPO:-"${DOCKER_ORG}/release"}
FINAL_IMAGE_REPO=${FINAL_IMAGE_REPO:-"${DOCKER_ORG}/ollama"}
OLLAMA_COMMON_BUILD_ARGS="--build-arg=VERSION \
    --build-arg=GOFLAGS \
    --build-arg=OLLAMA_CUSTOM_CPU_DEFS \
    --build-arg=OLLAMA_SKIP_CUDA_GENERATE \
    --build-arg=OLLAMA_SKIP_CUDA_11_GENERATE \
    --build-arg=OLLAMA_SKIP_CUDA_12_GENERATE \
    --build-arg=OLLAMA_SKIP_ROCM_GENERATE \
    --build-arg=AMDGPU_TARGETS"
OLLAMA_NEW_RUNNERS=${OLLAMA_NEW_RUNNERS:-""}
if [ -n "${OLLAMA_NEW_RUNNERS}" ]; then
    NEW_MAKEFILE=".new"
else
    NEW_MAKEFILE=""
fi

echo "Building Ollama version $VERSION"