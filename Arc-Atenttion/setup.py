# setup.py - Production Build for Intel oneAPI on Windows
"""
Production-grade build system optimized for:
- Intel oneAPI icpx (Clang-based) compiler
- Python 3.14 with venv support
- Battlemage (Xe2) hardware optimizations
- Memory-efficient compilation
"""

import sys
import os
import sysconfig
import subprocess
import platform
import shutil
from pathlib import Path
# Using pure distutils to avoid setuptools crash
from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext

class ProductionBuildExt(build_ext):
    """Production build extension with Intel compiler optimizations"""
    
    # __init__ removed to avoid interfering with distutils command instantiation

    
    def _validate_environment(self):
        """Validate build environment with detailed error reporting"""
        print("=" * 70)
        print("Intel Battlemage FlashAttention - Build Environment Validation")
        print("=" * 70)
        
        errors = []
        warnings = []
        
        # Check Python version and venv
        print(f"[INFO] Python: {sys.version}")
        print(f"[INFO] Prefix: {sys.prefix}")
        print(f"[INFO] Base Prefix: {sys.base_prefix}")
        
        # Check for venv
        if sys.prefix == sys.base_prefix:
            warnings.append("Not running in a virtual environment (venv)")
        else:
            print(f"[OK] Running in virtual environment: {sys.prefix}")
        
        # Check icpx compiler
        icpx_path = shutil.which("icpx")
        if not icpx_path:
            errors.append("Intel oneAPI compiler (icpx) not found in PATH")
        else:
            try:
                result = subprocess.run(
                    ["icpx", "--version"],
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
                if "Intel(R) oneAPI DPC++" in result.stdout:
                    print(f"[OK] Intel oneAPI compiler found: {icpx_path}")
                    
                    # Check for SYCL support
                    if "-fsycl" not in result.stdout:
                        warnings.append("Compiler may not have SYCL support")
                else:
                    errors.append("Intel oneAPI compiler not detected correctly")
            except Exception as e:
                errors.append(f"Failed to check compiler: {e}")
        
        include_dir = Path(sysconfig.get_path('include'))
        # Fix for Windows where LIBDIR might be None
        libs_dir_raw = sysconfig.get_config_var('LIBDIR')
        # Use base_prefix to find libs (venv doesn't have them)
        libs_dir = Path(libs_dir_raw) if libs_dir_raw else Path(sys.base_prefix) / "libs"
        
        if not include_dir.exists():
            errors.append(f"Python include directory not found: {include_dir}")
        else:
            print(f"[OK] Python includes: {include_dir}")
        
        if not libs_dir or not Path(libs_dir).exists():
            errors.append(f"Python library directory not found: {libs_dir}")
        else:
            print(f"[OK] Python libs: {libs_dir}")
            
            # List available libraries
            lib_files = list(Path(libs_dir).glob("python*.lib"))
            print(f"[INFO] Available libraries: {[f.name for f in lib_files]}")
            
            # Find the correct library
            python_version = f"{sys.version_info.major}{sys.version_info.minor}"
            target_lib = f"python{python_version}.lib"
            lib_path = Path(libs_dir) / target_lib
            
            if lib_path.exists():
                print(f"[OK] Target library found: {target_lib}")
            else:
                # Fallback: look for any python*.lib
                for lib in lib_files:
                    if 'd.lib' not in str(lib):  # Skip debug libs
                        print(f"[WARN] Using alternative library: {lib.name}")
                        break
        
        # Report errors and warnings
        if warnings:
            print("\n[WARNINGS]:")
            for warning in warnings:
                print(f"  - {warning}")
        
        if errors:
            print("\n[ERRORS]:")
            for error in errors:
                print(f"  - {error}")
            print("\n[FIX] Ensure:")
            print("  1. Intel oneAPI is installed and sourced")
            print("  2. Run: 'C:\\Program Files (x86)\\Intel\\oneAPI\\setvars.bat'")
            print("  3. Python development files are installed")
            sys.exit(1)
        
        print("\n" + "=" * 70)
        print("[SUCCESS] Environment validated successfully!")
        print("=" * 70)
    
    def build_extension(self, ext):
        """Build extension with Intel compiler optimizations"""
        # Validate environment only once (lazy check)
        if not getattr(self, "_env_validated", False):
            self._validate_environment()
            self._env_validated = True

        print(f"\n[BUILD] Building {ext.name}...")
        
        # Get system information
        system = platform.system()
        is_windows = system == "Windows"
        
        # Get Python paths (respecting venv)
        py_include = sysconfig.get_path('include')
        py_libs = sysconfig.get_config_var('LIBDIR')
        
        if not py_libs:
            # Fallback for venv
            py_libs = os.path.join(sys.prefix, "libs")
        
        # Output file
        ext_fullpath = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_fullpath), exist_ok=True)
        
        # Get Pybind11 include path
        try:
            import pybind11
            pybind11_include = pybind11.get_include()
        except ImportError:
            print("[ERROR] pybind11 not installed. Please run: pip install pybind11")
            sys.exit(1)

        
        # ====================================================================
        # Core Compiler Flags for Intel oneAPI (Clang-based)
        # ====================================================================
        
        # ====================================================================
        # Compiler Flags Configuration
        # ====================================================================
        
        # 1. Base Compiler & Standard
        base_flags = [
            "icpx",                     # Intel C++ Compiler
            "-O3",                      # Max optimization
            "-std=c++17",               # C++17 Standard
            "-fsycl",                   # SYCL Support
            "-fsycl-targets=spir64",    # Target SPIR-V 64-bit
            "-shared",
            f"-o{ext_fullpath}",
            ext.sources[0],
            f"-I{py_include}",
            f"-I{pybind11_include}",
            "-I.", # Allow local headers (e.g. oneapi/matrix/matrix-intel.hpp)
        ]

        # 2. Platform Specifics
        platform_flags = []
        if is_windows:
            platform_flags = [
                "-D_WIN32", "-D_WINDOWS", "-D_MBCS",
                # Flags removed per user feedback (icpx default handling)
                # "-MD", "-EHsc", "-Zc:__cplusplus",
                "-D_CRT_SECURE_NO_WARNINGS",
                "-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"
            ]
        else:
            platform_flags = [
                "-D__linux__",
                "-fPIC", "-fvisibility=hidden", "-march=native"
            ]

        # 3. Performance Optimizations (Aggressive)
        optim_flags = [
            "-ffast-math",              # Allow aggressive float optimizations
            "-fno-math-errno",          # No errno for math functions calls
            "-fno-trapping-math",       # Assume no floating point exceptions
            "-fno-alias",               # Assume strict aliasing (no pointer overlap unless specified)
            "-fno-builtin-malloc",      # Avoid compiler handling of malloc
            "-fno-builtin-free",
            "-funroll-loops",           # Unroll loops
            # "-flto",                    # Link Time Optimization (Disabled: requires lld)
        ]

        # 4. Debug & Safety (Production Mode)
        debug_flags = [
            "-DNDEBUG",                 # Disable asserts
        ]

        # 5. Dynamic Linker & Hardware Flags
        flags = base_flags + platform_flags + optim_flags + debug_flags
        
        if not py_libs or not os.path.exists(py_libs):
            # Fallback to base path (correct for venv)
            py_libs = os.path.join(sys.base_prefix, "libs")
            
        if py_libs and os.path.exists(py_libs):
            flags.append(f"-L{py_libs}") # Always add search path
            
            python_version = f"{sys.version_info.major}{sys.version_info.minor}"
            possible_libs = [
                f"python{python_version}.lib",
                "python3.lib"
            ]
            
            lib_found = False
            for lib_name in possible_libs:
                lib_path = os.path.join(py_libs, lib_name)
                if os.path.exists(lib_path):
                    flags.append(lib_path) # Absolute Path
                    print(f"[INFO] Linked Python library: {lib_path}")
                    lib_found = True
                    break
            
            if not lib_found:
                 print(f"[WARN] Could not find python lib in {py_libs}")
                 flags.append(f"-lpython{python_version}")

        # Hardware Features (Battlemage/Xe2)
        try:
             # Basic check to avoid errors if icpx is not responsive
             sys_check = subprocess.run(["icpx", "--version"], capture_output=True)
             if sys_check.returncode == 0:
                # Add Battlemage specific hints
                # flags.append("-Xsflags=-doubleGRF")
                # flags.append("-Xsflags=-no-local-scheduling")
                # flags.append("-Xsflags=-enable-xmx")
                print("[INFO] Skipped specific hardware hints (causing runtime 1104/api errors)")
        except Exception:
             pass
        
        # ====================================================================
        # Execute Build
        # ====================================================================
        
        # Remove empty flags
        flags = [f for f in flags if f]
        
        # Print truncated command for readability
        cmd_display = " ".join(flags[:10]) + "..." if len(flags) > 10 else " ".join(flags)
        print(f"[INFO] Build command: {cmd_display}")
        
        # Create build log directory
        log_dir = "build_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Execute with detailed logging
        try:
            print("[INFO] Starting compilation...")
            start_time = subprocess.run(
                ["icpx", "--version"],
                capture_output=True,
                text=True
            )
            print(f"[DEBUG] Compiler version:\n{start_time.stdout[:200]}...")
            
            # Run the actual build
            result = subprocess.run(
                flags,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Save logs (Robust)
            try:
                with open(os.path.join(log_dir, "build_output.log"), "w", encoding="utf-8", errors="replace") as f:
                    f.write(f"Command: {' '.join(flags)}\n")
                    f.write(f"Return code: {result.returncode}\n")
                    f.write(f"Stdout:\n{result.stdout}\n")
                    f.write(f"Stderr:\n{result.stderr}\n")
            except Exception as log_err:
                print(f"[WARN] Failed to write build logs: {log_err}")
            
            if result.returncode != 0:
                print(f"[ERROR] Build failed with code {result.returncode}")
                
                # Provide helpful error analysis
                if "LNK1181" in result.stderr:
                    print("\n[DIAGNOSIS] Linker error: Python library not found")
                    print("[SOLUTION] Check Python library path in venv")
                    print(f"  Expected in: {py_libs}")
                    if py_libs and os.path.exists(py_libs):
                        print(f"  Files present: {os.listdir(py_libs)}")
                
                elif "undefined reference" in result.stderr:
                    print("\n[DIAGNOSIS] Undefined symbols")
                    print("[SOLUTION] Check for missing library dependencies")
                
                elif "sycl" in result.stderr.lower():
                    print("\n[DIAGNOSIS] SYCL-related error")
                    print("[SOLUTION] Verify SYCL installation and -fsycl flag")
                
                print(f"\n[ERROR OUTPUT]:\n{result.stderr[:500]}...")
                sys.exit(1)
            
            print(f"[SUCCESS] Built {ext.name}")
            print(f"[INFO] Output: {ext_fullpath}")
            
            # Verify the built module
            if os.path.exists(ext_fullpath):
                file_size = os.path.getsize(ext_fullpath) / (1024 * 1024)
                print(f"[INFO] Module size: {file_size:.2f} MB")
            else:
                print("[WARN] Output file not found (but build reported success)")
        
        except subprocess.CalledProcessError as e:
            print(f"[FATAL] Build process failed: {e}")
            print(f"[DEBUG] Error output:\n{e.stderr[:500]}")
            sys.exit(1)
        
        except Exception as e:
            print(f"[FATAL] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

# ============================================================================
# Extension Configuration
# ============================================================================

ext_modules = [
    Extension(
        "attention_kernel_v3",
        sources=["attention_kernel_v3.cpp"],
        # Let the build class handle all compile/link args
    ),
    Extension(
        "attention_kernel_v5_dpas",
        sources=["attention_kernel_v5_dpas.cpp"],
        language="c++",
    )
]

# ============================================================================
# Setup Configuration
# ============================================================================

setup(
    name="attention_kernel",
    version="5.0.0",
    author="Intel GPU Optimization Team",
    description="Production FlashAttention for Intel Battlemage/Xe2",
    long_description="""
    High-performance FlashAttention implementation with:
    - Intel oneAPI DPC++ compiler optimizations
    - Battlemage (Xe2) hardware-specific tuning
    - Memory-efficient tiling algorithm (O(Nd) memory)
    - Production-grade error handling and validation
    """,
    # long_description_content_type="text/markdown", # Not supported in pure distutils
    ext_modules=ext_modules,
    cmdclass={"build_ext": ProductionBuildExt},
    
    # Package metadata
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
)