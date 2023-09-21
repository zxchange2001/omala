#!/bin/sh
# This script detects the current operating system architecture and installs the appropriate version of Ollama

set -eu

os=$(uname -s)
if [ "$os" != "Linux" ]; then
    echo "This script is intended to run on Linux only."
    exit 1
fi

# Determine the system architecture
ARCH=$(uname -m)

# Map architecture to the possible suffixes/names supported
case $ARCH in
    x86_64)
        ARCH_SUFFIX="amd64"
        ;;
    aarch64|arm64)
        ARCH_SUFFIX="arm64"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

if [ "$(id -u)" -ne 0 ]; then
    sudo_cmd="sudo "
    echo "Downloading the ollama executable to the PATH, this will require sudo permissions."
else
    sudo_cmd=""
fi

# Check if CUDA drivers are available
if command -v gcc >/dev/null 2>&1; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_VERSION=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*")
        if [ -z "$CUDA_VERSION" ]; then
            echo "Warning: NVIDIA-SMI is available, but the CUDA version cannot be detected. Installing CUDA drivers..."
            curl https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run | ${sudo_cmd}sh -s -- --silent --driver
        else
            echo "Detected CUDA version $CUDA_VERSION"
        fi
    else
        # Check for the presence of an NVIDIA GPU using lspci
        if lspci | grep -i "nvidia" >/dev/null 2>&1; then
            echo "Warning: NVIDIA GPU detected but NVIDIA-SMI is not available. Installing CUDA drivers..."
            curl https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run | ${sudo_cmd}sh -s -- --silent --driver
        else
            echo "No NVIDIA GPU detected. Skipping driver installation."
        fi
    fi
else
    echo "Warning: gcc is not installed. CUDA graphics driver requires gcc for installation. Ollama will run on CPU."
fi

${sudo_cmd}mkdir -p /usr/bin
${sudo_cmd}curl https://ollama.ai/download/latest/ollama-linux-$ARCH > /usr/bin/ollama

# Add ollama to start-up
if command -v systemctl >/dev/null 2>&1; then
    echo "Creating systemd service file for ollama..."
    cat <<EOF | ${sudo_cmd}tee /etc/systemd/system/ollama.service >/dev/null
[Unit]
Description=Ollama Service
After=network.target

[Service]
ExecStart=/usr/bin/ollama serve
Restart=always
RestartSec=3
Environment="HOME=$HOME"

[Install]
WantedBy=default.target
EOF
    echo "Reloading systemd and enabling ollama service..."
    ${sudo_cmd}systemctl daemon-reload
    ${sudo_cmd}systemctl enable ollama
    ${sudo_cmd}systemctl restart ollama
else
    echo "Installation complete. Run 'ollama serve' from the command line to start the service. Use 'ollama run' to query a model."
    exit 0
fi

echo "Installation complete. You can now run 'ollama' from the command line."
