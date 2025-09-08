#!/bin/bash

# Check if exactly 2 arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <source_folder> <destination_folder>"
    echo "Example: $0 folder1 folder2"
    exit 1
fi

SOURCE_DIR="$1"
DEST_DIR="$2"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# Create destination directory and data subdirectory
echo "Creating directories..."
mkdir -p "$DEST_DIR/data"

# Copy files from source root to destination root
echo "Copying files from $SOURCE_DIR to $DEST_DIR..."

# Copy *.py files
if ls "$SOURCE_DIR"/*.py 1> /dev/null 2>&1; then
    cp "$SOURCE_DIR"/*.py "$DEST_DIR/" 2>/dev/null
    echo "Copied *.py files"
fi

# Copy *.yaml files
if ls "$SOURCE_DIR"/*.yaml 1> /dev/null 2>&1; then
    cp "$SOURCE_DIR"/*.yaml "$DEST_DIR/" 2>/dev/null
    echo "Copied *.yaml files"
fi

# Copy *.sh files
if ls "$SOURCE_DIR"/*.sh 1> /dev/null 2>&1; then
    cp "$SOURCE_DIR"/*.sh "$DEST_DIR/" 2>/dev/null
    echo "Copied *.sh files"
fi

# Copy cans file (if it exists)
if [ -f "$SOURCE_DIR/cans" ]; then
    cp "$SOURCE_DIR/cans" "$DEST_DIR/"
    echo "Copied cans file"
fi

# Copy *.nml files
if ls "$SOURCE_DIR"/*.nml 1> /dev/null 2>&1; then
    cp "$SOURCE_DIR"/*.nml "$DEST_DIR/" 2>/dev/null
    echo "Copied *.nml files"
fi

# Copy files from source/data to destination/data (if source/data exists)
if [ -d "$SOURCE_DIR/data" ]; then
    echo "Copying files from $SOURCE_DIR/data to $DEST_DIR/data..."
    
    # Copy fld*.bin files
    if ls "$SOURCE_DIR/data"/fld*.bin 1> /dev/null 2>&1; then
        cp "$SOURCE_DIR/data"/fld*.bin "$DEST_DIR/data/" 2>/dev/null
        echo "Copied data/fld*.bin files"
    fi
    
    # Copy *.py files from data
    if ls "$SOURCE_DIR/data"/*.py 1> /dev/null 2>&1; then
        cp "$SOURCE_DIR/data"/*.py "$DEST_DIR/data/" 2>/dev/null
        echo "Copied data/*.py files"
    fi
    
    # Copy *.gnuplot files
    if ls "$SOURCE_DIR/data"/*.gnuplot 1> /dev/null 2>&1; then
        cp "$SOURCE_DIR/data"/*.gnuplot "$DEST_DIR/data/" 2>/dev/null
        echo "Copied data/*.gnuplot files"
    fi
    
    # Copy best_model file from checkpoint*
    if ls "$SOURCE_DIR/checkpoints_pettingzoo_grid_shared"/best_model.pt 1> /dev/null 2>&1; then
        cp "$SOURCE_DIR/checkpoints_pettingzoo_grid_shared"/best_model.pt "$DEST_DIR/checkpoints_pettingzoo_grid_shared/" 2>/dev/null
        echo "Copied best_model file"
    fi
    
    # Copy *.sh files from data
    if ls "$SOURCE_DIR/data"/*.sh 1> /dev/null 2>&1; then
        cp "$SOURCE_DIR/data"/*.sh "$DEST_DIR/data/" 2>/dev/null
        echo "Copied data/*.sh files"
    fi
else
    echo "Note: $SOURCE_DIR/data directory does not exist, skipping data files."
fi

echo "Project setup complete! Files copied from '$SOURCE_DIR' to '$DEST_DIR'"
