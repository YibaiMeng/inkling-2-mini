CXX := clang++
CXXFLAGS := -O3 -std=c++20 -Wall -Wextra -Wshadow -Wno-unused-parameter -Werror -fobjc-arc -march=native
LDFLAGS := -framework Metal -framework Foundation

BUILD_DIR := build
SRC_DIR := src

SRCS := $(SRC_DIR)/tokenizer.mm $(SRC_DIR)/model_weights.mm $(SRC_DIR)/prefix_cache.mm $(SRC_DIR)/engine.mm $(SRC_DIR)/http_server.mm $(SRC_DIR)/main.mm
HDRS := $(SRC_DIR)/json_lite.h $(SRC_DIR)/tokenizer.h $(SRC_DIR)/model_weights.h $(SRC_DIR)/prefix_cache.h $(SRC_DIR)/engine.h $(SRC_DIR)/http_server.h
OBJS := $(patsubst $(SRC_DIR)/%.mm,$(BUILD_DIR)/%.o,$(SRCS))

.PHONY: all clean lint format

all: $(BUILD_DIR)/inkling

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.mm $(HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/inkling: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) -o $@

lint:
	@echo "Running Clang static analyzer..."
	@for f in $(SRCS); do \
		$(CXX) --analyze -Xanalyzer -analyzer-output=text -std=c++20 -Wall -Wextra -Wshadow -Wno-unused-parameter -fobjc-arc $$f -o /dev/null || exit 1; \
	done
	@echo "Static analysis passed cleanly."

format:
	@command -v clang-format >/dev/null 2>&1 || { echo "clang-format not installed (brew install clang-format)"; exit 1; }
	clang-format -i $(SRCS) $(HDRS)

clean:
	rm -rf $(BUILD_DIR)
