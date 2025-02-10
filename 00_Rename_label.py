# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:52:52 2025

@author: thana
"""

import os

# 📌 กำหนดโฟลเดอร์ปลายทาง
dataset_folder = r"D:\University\3\3_2\Indus based\AGE_Detection\Datasets"

# 📌 วนลูปทุกโฟลเดอร์ใน `Datasets`
for class_folder in sorted(os.listdir(dataset_folder)):
    class_path = os.path.join(dataset_folder, class_folder)

    # ข้ามถ้าไม่ใช่โฟลเดอร์
    if not os.path.isdir(class_path):
        continue

    # 📌 ดึง Class Label และชื่อ Class
    class_label, class_name = class_folder.split("_", 1)  # เช่น "0_Child" → class_label = "0", class_name = "Child"

    # 📌 วนลูปทุกไฟล์ในโฟลเดอร์
    for i, filename in enumerate(sorted(os.listdir(class_path)), start=1):
        file_ext = filename.split(".")[-1].lower()  # ดึงนามสกุลไฟล์
        if file_ext not in ["jpg", "jpeg", "png"]:
            continue  # ข้ามถ้าไม่ใช่ไฟล์รูป

        # 📌 ตั้งชื่อใหม่ → 0_Child_0001.jpg, 0_Child_0002.jpg, ...
        new_filename = f"{class_label}_{class_name}_{i:04d}.{file_ext}"
        old_file = os.path.join(class_path, filename)
        new_file = os.path.join(class_path, new_filename)

        # 📌 เปลี่ยนชื่อไฟล์
        os.rename(old_file, new_file)
        print(f"✅ Rename: {filename} → {new_filename}")

print("🎯 Done! เปลี่ยนชื่อภาพเรียบร้อย ✅")


"""
if torch.cuda.is_available():
    print("Cuda is avialable")
else:
    print("No")
"""