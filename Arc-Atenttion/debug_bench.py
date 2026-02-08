import sys
import os

print("--- DEBUG START ---", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"CWD: {os.getcwd()}", flush=True)
print(f"PATH: {os.environ.get('PATH', '')[:200]}...", flush=True)

try:
    print("Importing attention_kernel...", flush=True)
    import attention_kernel
    print("attention_kernel imported.", flush=True)
except Exception as e:
    print(f"attention_kernel import failed: {e}", flush=True)

try:
    print("Importing dpctl...", flush=True)
    import dpctl
    print("dpctl imported.", flush=True)
    try:
        print("Selecting GPU...", flush=True)
        d = dpctl.SyclDevice("gpu")
        print(f"Device found: {d.name}", flush=True)
    except Exception as e:
        print(f"Device selection failed: {e}", flush=True)
except Exception as e:
    print(f"dpctl import failed: {e}", flush=True)

print("--- DEBUG END ---", flush=True)
