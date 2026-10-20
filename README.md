# FLAVRI-MCP

MCP Server for the [FLAVRI Multimodal Chemosensory Model](https://github.com/TheRedDeveloper/flavri). Exposes inference and optimization across FLAVRI's 148 labeled dimensions (11 taste + 137 odor channels).

## Quickstart

> [!TIP]
> **No GPU required!** For typical operations CPU execution is 2 to 5 times faster.

### Claude Desktop

```json
{
  "mcpServers": {
    "flavri": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "ghcr.io/thereddeveloper/flavri-mcp:latest"
      ]
    }
  }
}
```

---

### Other MCP Settings

- **Type:** `command`
- **Command:** `docker`
- **Arguments:** `run -i --rm ghcr.io/thereddeveloper/flavri-mcp:latest`

### GPU Accelerated / Server Hosting (Optional)

If running in a container environment with an NVIDIA GPU for high-throughput batching, use the `:cuda` tag with the `--gpus all` flag:

```json
{
  "mcpServers": {
    "flavri": {
      "command": "docker",
      "args": [
        "run",
        "--gpus",
        "all",
        "-i",
        "--rm",
        "ghcr.io/thereddeveloper/flavri-mcp:cuda"
      ]
    }
  }
}
```

### Self-Hosted (Network Daemon)

To run FLAVRI-MCP on your local network via SSE:

```bash
docker run -d \
  --name flavri-mcp \
  --restart always \
  -p 8000:8000 \
  ghcr.io/thereddeveloper/flavri-mcp:latest \
  --transport sse --host 0.0.0.0 --port 8000
```

Also available via Streamable HTTP (`--transport streamable-http`).

## Docker Image Variants

| Variant | Tag | Image Size | Description |
|:---|:---|:---|:---|
| **Standard (Lightweight)** | `:latest`, `:cpu` | **~1.2 GB** | Default. AVX2 CPU acceleration. |
| **CUDA (GPU Accelerated)** | `:cuda` | **~9.5 GB** | For humongous tasks. Full CUDA 12.1 runtime. |

## MCP Tools

| Tool | Description |
|:---|:---|
| `predict_flavor` | Predicts taste and odor for any single molecule or ratio mixture |
| `recommend_molecules` | Screens the 28,000+ molecule library for matching chemosensory notes |
| `modulate_flavor` | Screens additives to fill the remaining budget of an unnormalized base formula |
| `neutralize` | Solves continuous ratio optimization across 4 modalities: `"both"`, `"taste"`, `"odor"`, or `"evaporation"` |
| `balance_mixture` | Simplex optimization over candidate ingredients to achieve target profiles |
| `balance_perfume` | Multi-phase fine fragrance timeline optimizer across arbitrary hours |
| `scent_after` | Simulates fragrance evaporation decay over arbitrary time checkpoints |
| `list_descriptors` | Returns all 148 vocabulary descriptors (11 taste/trigeminal + 137 odor) |

## Local Development

If you wish to develop or run directly from Python:

```sh
git clone https://github.com/TheRedDeveloper/flavri-mcp
cd flavri-mcp

conda env create -f environment.yml
conda activate flavri-mcp
pip install -e .

# FIRST: Place flavri-0.8.pt into checkpoints/
python scripts/build_cache.py
```
