import os
import sys

BANNED_KEYWORDS = [
    "ffmpeg",
    "pyav",
    "av",
    "opencv",
    "cv2.videocapture",
    "videostream",
    "rtc.videostream",
]

def scan_file(file_path):
    violations = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line_lower = line.lower()
                for keyword in BANNED_KEYWORDS:
                    # Avoid false positives for imports like 'import asyncio' or 'import os' containing 'av' or similar substring.
                    # We check for exact substring boundaries or specific patterns.
                    if keyword in line_lower:
                        # Exclude self-references or harmless standard imports containing 'av'
                        if keyword == "av":
                            # Check if it's PyAV or av import/usage
                            if "import av" not in line_lower and "from av" not in line_lower and "av." not in line_lower:
                                continue
                        violations.append((line_no, keyword, line.strip()))
    except Exception as e:
        print(f"[WARNING] Could not read {file_path}: {e}")
    return violations

def run_audit():
    print("=" * 60)
    print("          VELO ZERO-CPU-DECODE PRODUCTION AUDIT          ")
    print("=" * 60)
    
    production_dirs = ["src/velo", "native/src"]
    all_violations = {}
    
    for p_dir in production_dirs:
        if not os.path.exists(p_dir):
            print(f"[WARNING] Directory {p_dir} does not exist. Skipping.")
            continue
            
        print(f"[INFO] Auditing production directory: {p_dir}")
        for root, dirs, files in os.walk(p_dir):
            for file in files:
                if file.endswith((".py", ".rs", ".cpp", ".h")):
                    file_path = os.path.join(root, file)
                    violations = scan_file(file_path)
                    if violations:
                        all_violations[file_path] = violations
                        
    if all_violations:
        print("\n[FAILED] Banned CPU decoding libraries or keywords found in production:")
        for file_path, violations in all_violations.items():
            print(f"\n  File: {file_path}")
            for line_no, keyword, content in violations:
                print(f"    Line {line_no:3d} (matched '{keyword}'): {content}")
        print("\nResult: FAILED")
        sys.exit(1)
    else:
        print("\n[SUCCESS] No banned CPU decoding keywords or libraries found in production code.")
        print("\nResult: VERIFIED")
        
if __name__ == "__main__":
    run_audit()
