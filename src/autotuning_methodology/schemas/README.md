# T4: Open Autotuning Results Schema

`T4.json` is a [JSON Schema](https://json-schema.org/draft/2020-12/schema) (Draft 2020-12)
describing the output format autotuning frameworks write their results in. A T4 document has
three top-level members:

| Member | Purpose |
| --- | --- |
| `schema_version` | Version of this schema the document follows, in `major.minor.patch` form. |
| `metadata` | Everything about the environment the tuning session ran in. |
| `results` | One entry per tested configuration: its parameters, timings and validity. |

## Metadata structure

The `metadata` object captures comprehensive reproducibility information organized into six
categories. **Every category, and every field within a category, is optional** — collect what
the machine can tell you and leave the rest out.

### Compute

What the kernels ran on, as the compute API reports it.

- `api` — compute API (`CUDA`, `OpenCL`, `HIP`, `SYCL`, `Vulkan`)
- `platform` — SDK/platform version, e.g. `NVIDIA CUDA 12.1`
- `device` — device name as reported by the API

### Framework

Which autotuner orchestrated the session.

- `autotuner` — framework name (`KernelTuner`, `KTT`, `CLTune`, `ATF`)
- `version` — framework version
- `backend` — backend implementation (`PyCUDA`, `CuPy`, `HIP`), if the framework has one

### Session

When the session ran and how its numbers should be read.

- `timestamp` — ISO 8601 UTC start time, e.g. `2024-03-15T14:30:00Z`
- `timeunit` — unit for the runtime measurements in `results`, one of `nanoseconds`,
  `microseconds`, `milliseconds`, `seconds`

### Hardware

Hardware inventory, stored as the raw JSON the introspection tools produce. The schema fixes
only each field's container type; the keys inside are whatever the tool printed, so a newer
tool version does not need a schema change.

- `accelerators` — array of GPU/accelerator entries, vendor-neutral so heterogeneous and
  multi-GPU hosts both fit. Each entry needs a `vendor` (`nvidia`, `amd`, `intel`, `apple`)
  and its `data`; `tool` names the query tool (`nvidia-smi`, `rocm-smi`, `clinfo`,
  `system_profiler`) that produced it.
- `cpu` — `lscpu --json` output (architecture, cores, cache)
- `memory` — `/proc/meminfo` parsed to JSON (total, free, swap)
- `storage` — `lsblk` array (devices, sizes, mount points)
- `system` — `lshw` hardware tree (motherboard, buses, peripherals)

### Environment

The software side of the same question.

- `os_info` — `lsb_release -a` output parsed to JSON (distribution, version, codename)
- `hostname` — system identifier, useful for multi-node runs
- `python_requirements` — array of requirement specifier strings, as `requirements.txt` holds
  them
- `software` — map of tool name to version (`gcc`, `nvcc`, `cuda-toolkit`, `cmake`, …). A value
  is normally a version string, but may be an object when a path or build flags are worth
  recording.

### DOI

- `doi` — Digital Object Identifier of the archived dataset, validated against the standard
  `10.{registrant}/{suffix}` form. Added after publishing to Zenodo, Figshare, OSF or an
  institutional archive, so whoever holds the file can find the archive it came from.

## Metadata collection workflow

### System metadata collection

A collector script (the reference implementation is `metadata.py` from
[odgaard/TuningSchema](https://github.com/odgaard/TuningSchema)) gathers the hardware and
environment data and emits JSON shaped like the two matching metadata categories:

```json
{
  "hardware": {
    "accelerators": [],
    "cpu": {},
    "memory": {},
    "storage": [],
    "system": {}
  },
  "environment": {
    "os_info": {},
    "hostname": "gpu-node-01",
    "python_requirements": [],
    "software": {}
  }
}
```

The collection commands behind those fields:

| Field | Command |
| --- | --- |
| `hardware.accelerators` | `nvidia-smi --query -x`, `rocm-smi`, `clinfo` → JSON |
| `hardware.cpu` | `lscpu --json` |
| `hardware.memory` | `cat /proc/meminfo` → JSON |
| `hardware.storage` | `lsblk -a` → JSON |
| `hardware.system` | `lshw -json -sanitize` |
| `environment.os_info` | `lsb_release -a` → JSON |
| `environment.hostname` | `hostname` |
| `environment.python_requirements` | `requirements.txt` |
| `environment.software` | `gcc --version`, `nvcc --version`, … |

**Privacy.** Strip the `<processes>` section from `nvidia-smi` output and always pass
`-sanitize` to `lshw` so serial numbers stay out of the file. Note that `hostname` can reveal
internal network topology, and absolute paths can reveal directory structure; document
whatever measures you took.

### Merging into T4 results

The autotuning framework merges the collected system metadata with the metadata it knows
itself (compute, framework, session):

**Python**

```python
import json

with open("system-metadata.json") as system_metadata_file:
    system_metadata = json.load(system_metadata_file)

output = {
    "schema_version": "1.1.0",
    "metadata": {
        "compute": {"api": "CUDA", "platform": "NVIDIA CUDA 12.1", "device": "NVIDIA A100-SXM4-40GB"},
        "framework": {"autotuner": "KernelTuner", "version": "1.0.0", "backend": "PyCUDA"},
        "session": {"timestamp": "2024-03-15T14:30:00Z", "timeunit": "milliseconds"},
        **system_metadata,  # adds hardware and environment
    },
    "results": [],
}
```

**C++ (nlohmann/json)**

```cpp
#include <nlohmann/json.hpp>
using json = nlohmann::json;

json system_metadata = /* load from file */;
json output = {
    {"schema_version", "1.1.0"},
    {"metadata", {
        {"compute", {{"api", "CUDA"}, {"platform", "NVIDIA CUDA 12.1"}, {"device", "NVIDIA A100-SXM4-40GB"}}},
        {"framework", {{"autotuner", "KTT"}, {"version", "1.0.0"}}},
        {"session", {{"timestamp", "2024-03-15T14:30:00Z"}, {"timeunit", "milliseconds"}}},
        {"hardware", system_metadata["hardware"]},
        {"environment", system_metadata["environment"]}
    }},
    {"results", json::array()}
};
```

## Validating a results file

```python
import json
from jsonschema import validate

with open("T4.json") as schema_file:
    schema = json.load(schema_file)
with open("my-results.json") as results_file:
    validate(instance=json.load(results_file), schema=schema)
```

`ajv` (JavaScript) and [jsonschemavalidator.net](https://www.jsonschemavalidator.net) work too.
