import xrayutilities as xu

try:
    print(f"CuKa1: {xu.wavelength('CuKa1')}")
    print(f"CuKa2: {xu.wavelength('CuKa2')}")
    print(f"Cu: {xu.wavelength('Cu')}")
except Exception as e:
    print(e)
