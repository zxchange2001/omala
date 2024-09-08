package gpu

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/klauspost/cpuid/v2"
	"golang.org/x/sys/cpu"
)

func GetCPUCapability() CPUCapability {
	if cpu.X86.HasAVX2 {
		return CPUCapabilityAVX2
	}
	if cpu.X86.HasAVX {
		return CPUCapabilityAVX
	}
	// else LCD
	return CPUCapabilityNone
}

func IsNUMA() bool {
	if runtime.GOOS != "linux" {
		// numa support in llama.cpp is linux only
		return false
	}
	ids := map[string]interface{}{}
	packageIds, _ := filepath.Glob("/sys/devices/system/cpu/cpu*/topology/physical_package_id")
	for _, packageId := range packageIds {
		id, err := os.ReadFile(packageId)
		if err == nil {
			ids[strings.TrimSpace(string(id))] = struct{}{}
		}
	}
	return len(ids) > 1
}

func IsIntelCoreUltraCpus() bool {
	return strings.Contains(cpuid.CPU.BrandName, "Core(TM) Ultra")
}
