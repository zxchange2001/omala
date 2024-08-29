package llama

// #include <stdlib.h>
// #include "llama.h"
import "C"

import (
	"fmt"
	"log/slog"
	"unsafe"
)

// SystemInfo is an unused example of calling llama.cpp functions using CGo
func SystemInfo() string {
	return C.GoString(C.llama_print_system_info())
}

type LoadedModel struct {
	model *C.struct_llama_model
}

func LoadModel(modelfile string, vocabOnly bool) (*LoadedModel, error) {
	// TODO figure out how to quiet down the logging so we don't have 2 copies of the model metadata showing up
	slog.Info("XXX initializing default model params")
	params := C.llama_model_default_params()
	params.vocab_only = C.bool(vocabOnly)

	cmodelfile := C.CString(modelfile)
	defer C.free(unsafe.Pointer(cmodelfile))

	slog.Info("XXX loading model", "model", modelfile)
	model := C.llama_load_model_from_file(cmodelfile, params)
	if model == nil {
		return nil, fmt.Errorf("failed to load model %s", modelfile)
	}
	return &LoadedModel{model}, nil
}

func FreeModel(model *LoadedModel) {
	C.llama_free_model(model.model)
}

func Tokenize(model *LoadedModel, content string) ([]int, error) {
	ccontent := C.CString(content)
	defer C.free(unsafe.Pointer(ccontent))

	tokenCount := len(content) + 2
	tokens := make([]C.int32_t, tokenCount)

	tokenCount = int(C.llama_tokenize(model.model, ccontent, C.int32_t(len(content)),
		&tokens[0], C.int32_t(tokenCount), true, true))
	if tokenCount < 0 {
		tokenCount = -tokenCount
		slog.Info("XXX got negative response", "count", tokenCount)
		tokens = make([]C.int32_t, tokenCount)
		tokenCount = int(C.llama_tokenize(model.model, ccontent, C.int32_t(len(content)), &tokens[0],
			C.int32_t(tokenCount), true, true))

		if tokenCount < 0 {
			return nil, fmt.Errorf("failed to tokenize: %d", tokenCount)
		}
	} else if tokenCount == 0 {
		return nil, nil
	}
	ret := make([]int, tokenCount)
	for i := range tokenCount {
		ret[i] = int(tokens[i])
	}
	slog.Debug("XXX tokenized", "tokens", tokens, "content", content)
	return ret, nil
}

func Detokenize(model *LoadedModel, tokens []int) string {
	slog.Info("XXX in CGO detokenize")
	var resp string
	for _, token := range tokens {
		buf := make([]C.char, 8)
		nTokens := C.llama_token_to_piece(model.model, C.int(token), &buf[0], 8, 0, true)
		if nTokens < 0 {
			buf = make([]C.char, -nTokens)
			nTokens = C.llama_token_to_piece(model.model, C.int(token), &buf[0], -nTokens, 0, true)
		}
		tokString := C.GoStringN(&buf[0], nTokens)
		resp += tokString
	}
	slog.Debug("XXX detokenized", "tokens", tokens, "content", resp)
	return resp
}
