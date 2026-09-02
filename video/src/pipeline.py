import subprocess
import sys
import os

def run_script(script_path):
    print(f"Running {script_path}...")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Error running {script_path} (return code {result.returncode})")
        sys.exit(result.returncode)
    print(f"Finished {script_path}\n")

def main():
    # Make sure we're in the video directory when running this
    # to maintain relative paths like 'output/...'
    src_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.dirname(src_dir)
    os.chdir(video_dir)

    scripts = [
        os.path.join("src", "VA.py"),
        os.path.join("src", "VC.py"),
        os.path.join("src", "VE.py")
    ]

    for script in scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"Warning: Could not find {script}")

if __name__ == "__main__":
    main()
