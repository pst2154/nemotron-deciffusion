"""Checkpoint identity shared by download, API responses, and health."""
import os

MODEL_ID = os.environ.get('MODEL_ID', 'nemotron-deciffusion/Nemotron-Deciffusion-8B-v2')
MODEL_NAME = MODEL_ID.rsplit('/', 1)[-1]
PINNED_REVISIONS = {
    'nemotron-deciffusion/Nemotron-Deciffusion-8B-v2': '1dea38015e86e101eb978a33fa493818fc71e366',
    'nemotron-deciffusion/Nemotron-Deciffusion-8B-v2-NVFP4': '162dab6ff218eb02696cd896b7f0a751fe910aff',
    'nvidia/Nemotron-Labs-Diffusion-8B': '16c67f0560b912e93e0cabb6e0c4f5c3086d95fc',
}

def model_revision():
    revision = os.environ.get('MODEL_REVISION') or PINNED_REVISIONS.get(MODEL_ID)
    if not revision:
        raise ValueError('Set MODEL_REVISION for an unrecognized MODEL_ID')
    return revision
