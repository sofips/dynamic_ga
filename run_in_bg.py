import os
import subprocess

nthreads = 1

# List of commands to run
commands = [ 
    #f"python3 exp_gen.py 512 n16_ps512_{nthreads}t_og {nthreads}" ,
    f"python3 exp_gen.py 1024 n16_ps1024_{nthreads}t_og {nthreads}" ,
    f"python3 exp_gen.py 2048 n16_ps2048_{nthreads}t_og {nthreads}",
    f"python3 exp_gen.py 4096 n16_ps4096_{nthreads}t_og {nthreads}",
    f"python3 exp_gen.py 8192 n16_ps1024_{nthreads}t_og {nthreads} ",
]

# Get the current working directory (script's location)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Loop through the commands
for cmd in commands:
    print(f"Starting: {cmd}")

    # Use subprocess to run the command
    process = subprocess.Popen(
        cmd,
        shell=True,  # Allows string-based commands
        cwd=script_dir,  # Ensures commands are run from the script's location
        stdout=subprocess.PIPE,  # Capture standard output
        stderr=subprocess.PIPE,  # Capture standard error
        universal_newlines=True  # Decode bytes to string
    )

    # Wait for the command to finish
    stdout, stderr = process.communicate()

    # Print the outputs
    print(f"Output of {cmd}:")
    print(stdout)
    if stderr:
        print(f"Error output of {cmd}:")
        print(stderr)

    # Check the exit code
    if process.returncode != 0:
        print(f"{cmd} failed with exit code {process.returncode}")
    else:
        print(f"{cmd} completed successfully.")

print("All programs completed.")
