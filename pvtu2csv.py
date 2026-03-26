import argparse
import numpy as np
import os
import pandas as pd
import pyvista as pv

def convert_pvd_to_csv(input_file, output_prefix, field_name, vector=False):
    """Reads a PVD file and saves data to CSV(s) using PyVista."""
    # Read the PVD file
    reader = pv.PVDReader(input_file)
    
    # Iterate through all time steps in the PVD file
    for i in range(len(reader.time_values)):
        reader.set_active_time_point(i)
        
        mesh = reader.read()[0]
      
        df = pd.DataFrame(mesh.points, columns=['X','Y', 'Z'])

        if vector:
            df[f'{field_name}_x'] = mesh[f'{field_name}'][:, 0]
            df[f'{field_name}_y'] = mesh[f'{field_name}'][:, 1]
            df[f'{field_name}_z'] = mesh[f'{field_name}'][:, 2]
        else:
            df[f'{field_name}'] = mesh[f'{field_name}'][:]

        # Construct output filename with time step
        output_file = f"{output_prefix}_{reader.time_values[i]}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ParaView .pvd file to .csv using PyVista")
    parser.add_argument("-i", "--input", required=True, help="Input .pvd file")
    parser.add_argument("-o", "--output", default="output", help="Prefix for output .csv files")
    parser.add_argument("--field_name", default="displacement", help="Name of field to extract")
    parser.add_argument("--vector", action='store_true', help="Specify if field is vector valued")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
    else:
        convert_pvd_to_csv(args.input, args.output, args.field_name, args.vector)
