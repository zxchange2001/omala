package gpu

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/ollama/ollama/format"
)

var CudartGlobs = []string{
	"/usr/local/cuda/lib64/libcudart.so*",
	"/usr/lib/x86_64-linux-gnu/nvidia/current/libcudart.so*",
	"/usr/lib/x86_64-linux-gnu/libcudart.so*",
	"/usr/lib/wsl/lib/libcudart.so*",
	"/usr/lib/wsl/drivers/*/libcudart.so*",
	"/opt/cuda/lib64/libcudart.so*",
	"/usr/local/cuda*/targets/aarch64-linux/lib/libcudart.so*",
	"/usr/lib/aarch64-linux-gnu/nvidia/current/libcudart.so*",
	"/usr/lib/aarch64-linux-gnu/libcudart.so*",
	"/usr/local/cuda/lib*/libcudart.so*",
	"/usr/lib*/libcudart.so*",
	"/usr/local/lib*/libcudart.so*",
}

var NvmlGlobs = []string{}

var NvcudaGlobs = []string{
	"/usr/local/cuda*/targets/*/lib/libcuda.so*",
	"/usr/lib/*-linux-gnu/nvidia/current/libcuda.so*",
	"/usr/lib/*-linux-gnu/libcuda.so*",
	"/usr/lib/wsl/lib/libcuda.so*",
	"/usr/lib/wsl/drivers/*/libcuda.so*",
	"/opt/cuda/lib*/libcuda.so*",
	"/usr/local/cuda/lib*/libcuda.so*",
	"/usr/lib*/libcuda.so*",
	"/usr/local/lib*/libcuda.so*",
}

var OneapiGlobs = []string{
	"/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so*",
	"/usr/lib*/libze_intel_gpu.so*",
}

var (
	CudartMgmtName = "libcudart.so*"
	NvcudaMgmtName = "libcuda.so*"
	NvmlMgmtName   = "" // not currently wired on linux
	OneapiMgmtName = "libze_intel_gpu.so*"
)

func GetCPUMem() (memInfo, error) {
	var mem memInfo
	var total, available, free, buffers, cached, freeSwap uint64
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return mem, err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := s.Text()
		switch {
		case strings.HasPrefix(line, "MemTotal:"):
			_, err = fmt.Sscanf(line, "MemTotal:%d", &total)
		case strings.HasPrefix(line, "MemAvailable:"):
			_, err = fmt.Sscanf(line, "MemAvailable:%d", &available)
		case strings.HasPrefix(line, "MemFree:"):
			_, err = fmt.Sscanf(line, "MemFree:%d", &free)
		case strings.HasPrefix(line, "Buffers:"):
			_, err = fmt.Sscanf(line, "Buffers:%d", &buffers)
		case strings.HasPrefix(line, "Cached:"):
			_, err = fmt.Sscanf(line, "Cached:%d", &cached)
		case strings.HasPrefix(line, "SwapFree:"):
			_, err = fmt.Sscanf(line, "SwapFree:%d", &freeSwap)
		default:
			continue
		}
		if err != nil {
			return mem, err
		}
	}
	mem.TotalMemory = total * format.KibiByte
	mem.FreeSwap = freeSwap * format.KibiByte
	if available > 0 {
		mem.FreeMemory = available * format.KibiByte
	} else {
		mem.FreeMemory = (free + buffers + cached) * format.KibiByte
	}

	//Don'r try to load model to RAM that is already used By GTT
	amdGPUs := AMDGetGPUInfo()
	for _, gpuInfo := range amdGPUs {
		if gpuInfo.ApuUseGTT {
			mem.TotalMemory -= gpuInfo.TotalMemory
			mem.FreeMemory -= gpuInfo.TotalMemory
		}
	}
	return mem, nil
}
