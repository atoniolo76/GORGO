import modal
import time

app = modal.App(name="GORGO")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "cmake",
        "findutils",
        "libclang-dev",
        "libc-dbg",
        "libglib2.0-0",
        "libglib2.0-dev",
        "make",
        "netbase",
        "python3-networkx",
        "xz-utils",
        "util-linux",
        "gcc",
        "g++",
        "curl",
    )
    .apt_install("git")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({"PATH": "/root/.cargo/bin:/root/.local/bin:$PATH"})
    .run_commands(
        "mkdir -m 0755 /nix && chown root /nix",
        "groupadd -r nixbld && for i in $(seq 1 10); do useradd -r -g nixbld -G nixbld -d /var/empty -s /sbin/nologin nixbld$i; done",
        "mkdir -p /etc/nix && echo 'build-users-group =' > /etc/nix/nix.conf",
        "curl -L https://nixos.org/nix/install | sh -s -- --no-daemon",
    )
    .env({"PATH": "/root/.cargo/bin:/root/.local/bin:/root/.nix-profile/bin:/nix/var/nix/profiles/default/bin:$PATH"})
    .run_commands(
        "git clone https://github.com/shadow/shadow.git /opt/shadow",
        "cd /opt/shadow && ./setup build",
        "cd /opt/shadow && ./setup install",
    )
)

@app.function(image=image, timeout=3600, gpu="A10G")
def model_endpoint():
    import subprocess
    result = subprocess.run(["nix-shell", "-p", "git", "--run", "git"], capture_output=True, text=True)
    print(result.stdout)
    return result.stdout

if __name__ == "__main__":
    model_endpoint.remote();