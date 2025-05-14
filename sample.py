import os

gt_folder = r"C:\Users\hp\Desktop\Crowd Density\dataset\part_A_final\train_data\ground_truth"
mat_files = os.listdir(gt_folder)

print(mat_files[:10])  # Print first 10 files
