#!/usr/bin/env python3
"""
Intel Battlemage FlashAttention - Optimized Build Script
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def validate_environment():
    """Validate build environment and dependencies"""
    required_files = [
        "attention_kernel.cpp",
        # "sycl/sycl.hpp", # Headers are in include paths, not local
        # "sycl/ext/intel/esimd.hpp"
    ]
    
    # Check local files
    for file in required_files:
        if not Path(file).exists():
            print(f"[ERROR] Missing required file: {file}")
            return False
            
    # Check compiler
    ICPX_PATH = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\icpx.EXE"
    if not Path(ICPX_PATH).exists():
         print(f"[ERROR] Compiler not found at: {ICPX_PATH}")
         return False
    
    return True

def get_compiler_flags():
    """Get optimized compiler flags for Battlemage"""
    return [
        '-O3',                    # Maximum optimization
        '-std=c++17',            # C++17 standard
        '-fsycl',                # Enable SYCL
        '-fsycl-targets=spir64', # Target SPIR64
        '-fno-sycl-early-optimizations',
        '-Xclang', '-fsycl-allow-func-ptr',
        '-Wno-unknown-cuda-version',
        '-Wno-deprecated-declarations',
        
        # Intel-specific optimizations
        '-mllvm', '-force-vector-width=16',
        '-mllvm', '-enable-load-pre',
        
        # [DPA] Large Register File (256 regs)
        # This allows pre-fetching many tiles without spilling.
        '-Xs', '"-doubleGRF"',
        
        # Numeric stability
        '-ffast-math',
        '-fno-math-errno',
        
        # Debug info (optional)
        # '-g',
        # '-gdwarf-4',
    ]

def main():
    PROJECT_DIR = Path(__file__).parent
    OUTPUT_FILE = PROJECT_DIR / "attention_kernel.pyd"
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Clean previous build
    print("[CLEAN] Removing previous builds...")
    # Explicitly list extensions to delete to avoid deleting source code (attention_kernel.cpp)
    extensions_to_delete = ['.pyd', '.exp', '.lib', '.obj']
    
    # Remove build directory
    if (PROJECT_DIR / "build").exists():
        shutil.rmtree(PROJECT_DIR / "build")
        
    # Remove artifacts in project root
    for f in PROJECT_DIR.glob("attention_kernel.*"):
        if f.suffix in extensions_to_delete:
            try:
                f.unlink(missing_ok=True)
                print(f"  Deleted {f.name}")
            except Exception as e:
                print(f"  Failed to delete {f.name}: {e}")
    
    # Build directory
    BUILD_DIR = PROJECT_DIR / "build"
    BUILD_DIR.mkdir(exist_ok=True)
    
    # Compiler paths - User configuration
    ICPX_PATH = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\icpx.EXE"
    
    # Dynamic Path Resolution
    import sysconfig
    import pybind11
    
    PYTHON_INCLUDE = sysconfig.get_path("include")
    PYBIND11_INCLUDE = pybind11.get_include()
    
    # Windows-specific: Find python lib
    # Usually in <base>/libs/pythonXY.lib
    base_prefix = Path(sys.base_prefix)
    major = sys.version_info.major
    minor = sys.version_info.minor
    PYTHON_LIB = base_prefix / "libs" / f"python{major}{minor}.lib"
    
    if not PYTHON_LIB.exists():
         # Fallback for Python 3.14 if needed, though the above should work
         PYTHON_LIB = base_prefix / "libs" / "python314.lib"

    print(f"[CONFIG] Python Include: {PYTHON_INCLUDE}")
    print(f"[CONFIG] Pybind11 Include: {PYBIND11_INCLUDE}")
    print(f"[CONFIG] Python Lib: {PYTHON_LIB}")

    # Library paths to resolve libmmd.lib and others
    INTEL_LIB_DIR = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib"
    
    # Compilation flags
    compile_flags = get_compiler_flags()
    
    # Build command
    cmd = [
        f'"{ICPX_PATH}"',
        *compile_flags,
        f'-I"{PYTHON_INCLUDE}"',
        f'-I"{PYBIND11_INCLUDE}"',
        f'-L"{INTEL_LIB_DIR}"',
        f'"{PROJECT_DIR / "attention_kernel.cpp"}"',
        f'-o"{OUTPUT_FILE}"',
        f'"{PYTHON_LIB}"',
        '-shared',
    ]
    
    # Execute build
    print(f"[BUILD] Compiling with {len(cmd)} arguments...")
    full_cmd = ' '.join(cmd)
    # print(full_cmd) # Debug
    
    try:
        result = subprocess.run(
            full_cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Check output file
        if OUTPUT_FILE.exists():
            size_kb = OUTPUT_FILE.stat().st_size / 1024
            print(f"[SUCCESS] Built {OUTPUT_FILE}")
            print(f"         Size: {size_kb:.2f} KB")
            return True
        else:
            print("[ERROR] Build succeeded but output file not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Build failed with code {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)