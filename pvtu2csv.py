import argparse
import numpy as np
import os
import pandas as pd
import pyvista as pv

def convert_pvd_to_csv(input_file, output_prefix):
    """Reads a PVD file and saves data to CSV(s) using PyVista."""
    # Read the PVD file
    reader = pv.PVDReader(input_file)
    
    # Iterate through all time steps in the PVD file
    for i in range(len(reader.time_values)):
        reader.set_active_time_point(i)
        
        mesh = reader.read()[0]
      
        df = pd.DataFrame(mesh.points, columns=['X','Y', 'Z'])
        df['displacement_x'] = mesh['displacement'][:, 0]
        df['displacement_y'] = mesh['displacement'][:, 1]
        df['displacement_z'] = mesh['displacement'][:, 2]

        # Construct output filename with time step
        output_file = f"{output_prefix}_{i}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ParaView .pvd file to .csv using PyVista")
    parser.add_argument("-i", "--input", required=True, help="Input .pvd file")
    parser.add_argument("-o", "--output", default="output", help="Prefix for output .csv files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
    else:
        convert_pvd_to_csv(args.input, args.output)

