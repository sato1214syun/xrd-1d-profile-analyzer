import xrayutilities as xu
import numpy as np
from pathlib import Path
import sys

# Add src to path to allow imports if needed (though this test uses xu directly)
sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_inspect_crystal():
    # Use relative path from the test file location
    # Assuming running from root, but let's be robust
    base_dir = Path(__file__).parent.parent
    cif_path = base_dir / "data/cif/1538066.cif"

    if not cif_path.exists():
        print(f"CIF file not found at {cif_path}")
        return

    print(f"Loading CIF from: {cif_path}")
    crystal = xu.materials.Crystal.fromCIF(str(cif_path))

    print("Crystal object methods:")
    # print([d for d in dir(crystal) if not d.startswith("_")])

    # Get lattice
    lattice = crystal.lattice
    print(f"\nLattice: {lattice}")

    # Get allowed HKLs
    q_max = 4.0 * np.pi / 1.5406 * np.sin(np.radians(90 / 2))  # approx for 2theta=90
    hkls = lattice.get_allowed_hkl(q_max)
    print(f"\nNumber of HKLs: {len(hkls)}")

    # Try to calculate Structure Factor
    if len(hkls) > 0:
        hkl = list(hkls)[0]
        print(f"\nTesting HKL: {hkl}")

        # Calculate Q for this HKL
        Q = lattice.GetQ(hkl)
        print(f"Q vector: {Q}")

        # Calculate Structure Factor
        try:
            F = crystal.StructureFactor(Q)
            print(f"Structure Factor F: {F}")
            print(f"|F|^2: {np.abs(F) ** 2}")
        except Exception as e:
            print(f"Error calculating StructureFactor: {e}")

        # Check for multiplicity
        # Usually handled by symmetry
        # print(f"Symmetry: {crystal.symmetry}")
        # print([d for d in dir(crystal.symmetry) if not d.startswith("_")])


if __name__ == "__main__":
    test_inspect_crystal()
