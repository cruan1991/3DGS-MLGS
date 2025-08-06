from plyfile import PlyData
import sys

def inspect_ply(ply_path):
    plydata = PlyData.read(ply_path)
    print(f"📦 PLY File: {ply_path}")
    print(f"🔍 Elements:")
    for element in plydata.elements:
        print(f"  🧱 Element: {element.name} (count: {element.count})")
        print("    Fields:")
        for prop in element.properties:
            print(f"      - {prop.name} ({prop.dtype})")
    
    print("\n🔎 Sample Data (first 3 rows):")
    vertex_data = plydata['vertex']
    for i in range(min(3, len(vertex_data))):
        print(dict(vertex_data[i]))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_ply.py path/to/gaussians.ply")
    else:
        inspect_ply(sys.argv[1])
