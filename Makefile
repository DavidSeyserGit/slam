# Define the compiler and compiler flags
CXX = g++
CXXFLAGS = -std=c++11 -g `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`

# Define the target executable
TARGET = VideoCaptureMatches

# Define the source files
SRCS = slam.cpp

# Define the object files
OBJS = $(SRCS:.cpp=.o)

# Default target to build the executable
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to build the object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up the build artifacts
clean:
	rm -f $(TARGET) $(OBJS)

# Phony targets
.PHONY: all clean
