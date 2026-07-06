# hash:sha256:5a5b52110e8f6ba05d057f8341cd589a477f82af571e7b4b13c00b455429ac3f
FROM registry.codeocean.com/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN conda install -y -n base -c conda-forge \
        python=3.12 pip \
        c-compiler cxx-compiler make cmake git rust libboost-devel \
    && conda clean -afy

COPY requirements.txt /tmp/requirements.txt
RUN pip install torch==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && pip install -r /tmp/requirements.txt \
    && pip install --no-deps "bootstrapper @ git+https://github.com/ucsdmanorlab/bootstrapper.git@v0.3.2"

RUN python - <<'PY'
from packaging.version import Version
import zarr, funlib.persistence as fp
assert Version(zarr.__version__) < Version("3"), f"zarr too new: {zarr.__version__}"
assert Version(fp.__version__) < Version("0.7"), f"funlib.persistence too new: {fp.__version__}"
import bootstrapper, gunpowder, daisy, mwatershed, waterz  # noqa
print("environment ok: zarr", zarr.__version__, "funlib.persistence", fp.__version__)
PY
