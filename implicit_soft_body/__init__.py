import os

IMPLICIT_SOFT_BODY_ROOT = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(IMPLICIT_SOFT_BODY_ROOT, "..", "assets", "mesh")):
    IMPLICIT_SOFT_BODY_MESH_ROOT = os.path.join(IMPLICIT_SOFT_BODY_ROOT, "..", "assets", "mesh")
else:
    IMPLICIT_SOFT_BODY_MESH_ROOT = None

if os.path.exists(os.path.join(IMPLICIT_SOFT_BODY_ROOT, "..", "assets", "policy")):
    IMPLICIT_SOFT_BODY_POLICY_ROOT = os.path.join(IMPLICIT_SOFT_BODY_ROOT, "..", "assets", "policy")
else:
    IMPLICIT_SOFT_BODY_POLICY_ROOT = None