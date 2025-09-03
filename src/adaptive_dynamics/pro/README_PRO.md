# Adaptive Dynamics Toolkit Pro

This directory contains the Pro/commercial features of the Adaptive Dynamics Toolkit.

## Access Restrictions

The code in this directory is **NOT** covered by the MIT license that applies to the rest of the ADT codebase. These features require a commercial license for use in production environments.

## Feature Overview

The Pro version includes advanced features such as:

- CUDA kernels for GPU-accelerated compression
- Advanced 3D slicer heuristics
- Specialized beamline and gravitational solvers
- Enterprise dashboards and visualization tools
- Premium physics modules with higher precision

## Obtaining Access

To gain access to these features, contact us at RDM3DC@outlook.com to inquire about commercial licensing options.

## Integration

The Pro features integrate seamlessly with the open-source components of ADT through a consistent API design. When properly licensed, the Pro features can be imported just like any other ADT module:

```python
# Regular import for open-source components
from adaptive_dynamics.pi import AdaptivePi

# Pro feature import (requires valid license)
from adaptive_dynamics.pro.compress import GPUCompressor
```

## License Validation

Pro features use a license validation system that checks for a valid license key at runtime. Without a valid license, attempting to use Pro features will raise a `LicenseError`.

## Support

Commercial licensees receive priority support through our dedicated support portal. Please contact RDM3DC@outlook.com with any technical issues related to Pro features.
