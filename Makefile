CU=/opt/cuda/bin/nvcc
CC=clang++
LIBS=-lcublas
CPP_SOURCE=./src/cpp
HPP_SOURCE=./src/cpp/include
CUDA_SOURCE=./src/cuda
TEST_SOURCE=./test
INCLUDE_DIR=-I./src/cuda/include -I./src/cpp/include -I./src/cuda/
BUILD=./build
BIN=./bin
MAIN_SOURCE=./benchmark
STD=c++17
FLAGS=-gencode=arch=compute_35,code=sm_35 \
    -gencode=arch=compute_50,code=sm_50 \
    -gencode=arch=compute_52,code=sm_52 \
    -gencode=arch=compute_60,code=sm_60 \
    -gencode=arch=compute_61,code=sm_61 \
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_75,code=sm_75 \
    -gencode=arch=compute_60,code=compute_60
OPTI=-O3
DEBUG=--debug -g -G -O0

Wno=-Xcudafe "--diag_suppress=declared_but_not_referenced" -Wno-deprecated-gpu-targets

$(BUILD)/%.o: $(CPP_SOURCE)/%.cpp 
	$(CC) -std=$(STD) $(OPTI) $(INCLUDE_DIR) -c $< -o $@

$(BUILD)/%.o: $(HPP_SOURCE)/%.hpp 
	$(CC) -std=$(STD) $(OPTI) $(INCLUDE_DIR) -c $< -o $@

$(BUILD)/%.o: $(TEST_SOURCE)/%.cpp
	$(CC) -std=$(STD) $(OPTI) $(INCLUDE_DIR) -c $< -o $@

$(BUILD)/%.o: $(TEST_SOURCE)/%.cu
	$(CU) -std=$(STD) $(OPTI) $(INCLUDE_DIR) -c $< -o $@  $(FLAGS) $(Wno)

$(BUILD)/%.o: $(CUDA_SOURCE)/%.cu
	$(CU) -std=$(STD) $(OPTI) $(INCLUDE_DIR) -c $< -o $@  $(FLAGS) $(Wno)

$(BUILD)/%.o: $(MAIN_SOURCE)/%.cu $(DEP)
	$(CU) -std=$(STD) $(OPTI) $(INCLUDE_DIR) -c $< -o $@  $(FLAGS) $(Wno)

$(BUILD)/%-d.o: $(CPP_SOURCE)/%.cpp 
	$(CC) -std=$(STD) $(DEBUG) $(INCLUDE_DIR) -c $< -o $@

$(BUILD)/%-d.o: $(HPP_SOURCE)/%.hpp 
	$(CC) -std=$(STD) $(DEBUG) $(INCLUDE_DIR) -c $< -o $@

$(BUILD)/%-d.o: $(TEST_SOURCE)/%.cpp
	$(CC) -std=$(STD) $(DEBUG) $(INCLUDE_DIR) -c $< -o $@

$(BUILD)/%-d.o: $(TEST_SOURCE)/%.cu
	$(CU) -std=$(STD) $(DEBUG) $(INCLUDE_DIR) -c $< -o $@  $(FLAGS) $(Wno)

$(BUILD)/%-d.o: $(CUDA_SOURCE)/%.cu
	$(CU) -std=$(STD) $(DEBUG) $(INCLUDE_DIR) -c $< -o $@  $(FLAGS) $(Wno)

$(BUILD)/%-d.o: $(MAIN_SOURCE)/%.cu $(DEP)
	$(CU) -std=$(STD) $(DEBUG) $(INCLUDE_DIR) -c $< -o $@  $(FLAGS) $(Wno)

benchmark_%: $(BUILD)/benchmark_%.o $(BUILD)/%_gemm.o
	mkdir -p $(BIN)
	$(CU) $^ -std=$(STD) $(OPTI) -o $(BIN)/$@ $(LIBS) $(FLAGS) $(Wno)
	# sh ${SCRIPT_SOURCE}/$@.sh

test_%: $(BUILD)/test.o $(BUILD)/%_gemm.o
	$(CU) $^ -std=$(STD) -o $(BIN)/$@ -g $(LIBS) $(FLAGS) $(Wno)

test_%-d: $(BUILD)/test-d.o $(BUILD)/%_gemm-d.o
	$(CU) $^ -std=$(STD) -o $(BIN)/$@ -g $(LIBS) $(FLAGS) $(Wno)

.PHONY: clean
clean:
	rm $(BUILD)/*